"""
Robust Input Validation & Safe Inference Gate for Alzheimer's MRI Analysis.
Performs multi-stage file integrity, image quality, and conservative suitability screening.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import io
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
MIN_DIMENSION = 32
MAX_DIMENSION = 8192
RECOMMENDED_MIN_DIM = 64


class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    REJECTED = "REJECTED"


class ValidationCheck:
    """Individual verification check result."""
    def __init__(self, name: str, passed: bool, status: ValidationStatus, message: str):
        self.name = name
        self.passed = passed
        self.status = status
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "status": self.status.value,
            "message": self.message
        }


class ValidationResult:
    """Consolidated validation summary with structured checks and metadata."""
    def __init__(
        self,
        status: ValidationStatus,
        headline: str,
        message: str,
        checks: List[ValidationCheck],
        metrics: Dict[str, Any],
        sanitized_image: Optional[Image.Image] = None
    ):
        self.status = status
        self.is_valid = (status != ValidationStatus.REJECTED)
        self.can_run_inference = self.is_valid
        self.headline = headline
        self.message = message
        self.checks = checks
        self.metrics = metrics
        self.sanitized_image = sanitized_image

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "can_run_inference": self.can_run_inference,
            "headline": self.headline,
            "message": self.message,
            "checks": [c.to_dict() for c in self.checks],
            "metrics": self.metrics
        }


def validate_input_image(
    image_source: Union[Image.Image, bytes, io.BytesIO, str, Path],
    filename: Optional[str] = None
) -> ValidationResult:
    """
    Execute comprehensive multi-stage validation and conservative suitability screening.
    
    Stages:
    1. File & Decoding Check: Format, decodability, corruption check.
    2. Resolution & Geometry: Bounds, extreme aspect ratios.
    3. Signal & Uniformity: Blank (solid black/white) and near-uniform pixel check.
    4. Suitability Screening: Chromatic saturation and structural contrast check.
    5. Sanitization: Safe conversion of RGBA/Grayscale into clean 3-channel RGB.
    """
    checks: List[ValidationCheck] = []
    metrics: Dict[str, Any] = {}
    
    # -------------------------------------------------------------
    # Stage 1: File Loading & Image Decoding
    # -------------------------------------------------------------
    pil_img: Optional[Image.Image] = None
    original_format = "UNKNOWN"
    
    try:
        if isinstance(image_source, Image.Image):
            pil_img = image_source
            original_format = pil_img.format or "RAW_PIL"
        elif isinstance(image_source, (bytes, bytearray)):
            if len(image_source) == 0:
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    headline="Empty File",
                    message="Uploaded file contains 0 bytes.",
                    checks=[ValidationCheck("File Integrity", False, ValidationStatus.REJECTED, "File has 0 bytes.")],
                    metrics={}
                )
            bio = io.BytesIO(image_source)
            pil_img = Image.open(bio)
            original_format = pil_img.format or "UNKNOWN"
            pil_img.load()  # Force decode to detect corrupted data
        elif isinstance(image_source, io.BytesIO):
            image_source.seek(0)
            pil_img = Image.open(image_source)
            original_format = pil_img.format or "UNKNOWN"
            pil_img.load()
        elif isinstance(image_source, (str, Path)):
            p = Path(image_source)
            if not p.exists() or p.stat().st_size == 0:
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    headline="File Missing or Empty",
                    message=f"File at {p} could not be found or is empty.",
                    checks=[ValidationCheck("File Integrity", False, ValidationStatus.REJECTED, "File not found.")],
                    metrics={}
                )
            pil_img = Image.open(p)
            original_format = pil_img.format or "UNKNOWN"
            pil_img.load()
        else:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                headline="Unsupported Input Type",
                message="Input source is not a recognized file, buffer, or PIL Image.",
                checks=[ValidationCheck("Input Type", False, ValidationStatus.REJECTED, "Invalid type.")],
                metrics={}
            )
            
        # Format check
        norm_format = original_format.upper()
        if norm_format not in SUPPORTED_FORMATS and norm_format != "RAW_PIL":
            # Also check extension if format was ambiguous
            ext = Path(filename).suffix.lower().lstrip(".") if filename else ""
            if ext not in {"jpg", "jpeg", "png", "webp"}:
                checks.append(ValidationCheck(
                    "Format Support",
                    False,
                    ValidationStatus.REJECTED,
                    f"Format '{original_format}' is not supported. Please upload JPEG, PNG, or WEBP."
                ))
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    headline="Unsupported Image Format",
                    message=f"Image format '{original_format}' is not supported. Please upload JPEG, PNG, or WEBP.",
                    checks=checks,
                    metrics={"format": original_format}
                )
                
        checks.append(ValidationCheck(
            "File & Decode Integrity",
            True,
            ValidationStatus.PASS,
            f"Successfully decoded valid {original_format} image."
        ))
        
    except Exception as decode_err:
        checks.append(ValidationCheck(
            "File & Decode Integrity",
            False,
            ValidationStatus.REJECTED,
            f"Image could not be decoded: {decode_err}"
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Corrupted or Unreadable Image",
            message="The uploaded image file is corrupted or contains invalid image data.",
            checks=checks,
            metrics={"decode_error": str(decode_err)}
        )

    # -------------------------------------------------------------
    # Stage 2: Resolution & Aspect Ratio Screening
    # -------------------------------------------------------------
    width, height = pil_img.size
    aspect_ratio = width / max(height, 1)
    
    metrics["width"] = width
    metrics["height"] = height
    metrics["aspect_ratio"] = round(aspect_ratio, 2)
    metrics["original_mode"] = pil_img.mode
    metrics["original_format"] = original_format
    
    # Reject if absurdly tiny
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        checks.append(ValidationCheck(
            "Resolution Check",
            False,
            ValidationStatus.REJECTED,
            f"Resolution ({width}×{height}) is below minimum threshold ({MIN_DIMENSION}×{MIN_DIMENSION} px)."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Resolution Too Small",
            message="Image resolution is too small for reliable convolutional processing.",
            checks=checks,
            metrics=metrics
        )
        
    # Reject if absurdly huge
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        checks.append(ValidationCheck(
            "Resolution Check",
            False,
            ValidationStatus.REJECTED,
            f"Resolution ({width}×{height}) exceeds maximum supported size ({MAX_DIMENSION}×{MAX_DIMENSION} px)."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Resolution Exceeds Limits",
            message=f"Image dimensions ({width}×{height}) exceed processing limits.",
            checks=checks,
            metrics=metrics
        )

    # Extreme aspect ratio check (axial brain MRIs are near-square)
    aspect_check_passed = True
    aspect_status = ValidationStatus.PASS
    aspect_msg = f"Aspect ratio ({aspect_ratio:.2f}:1) is standard."
    
    if aspect_ratio > 4.0 or aspect_ratio < 0.25:
        aspect_check_passed = False
        aspect_status = ValidationStatus.REJECTED
        aspect_msg = f"Extreme aspect ratio ({aspect_ratio:.2f}:1) is unsuitable for axial MRI analysis."
        checks.append(ValidationCheck("Aspect Ratio", False, aspect_status, aspect_msg))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Unusable Aspect Ratio",
            message="The image is an extreme horizontal or vertical strip and cannot represent an axial brain MRI slice.",
            checks=checks,
            metrics=metrics
        )
    elif aspect_ratio > 2.2 or aspect_ratio < 0.45:
        aspect_check_passed = False
        aspect_status = ValidationStatus.WARNING
        aspect_msg = f"Unusual aspect ratio ({aspect_ratio:.2f}:1). Standard brain MRI scans have near-square proportions."
        
    checks.append(ValidationCheck("Dimensions & Geometry", aspect_check_passed, aspect_status, aspect_msg))

    # -------------------------------------------------------------
    # Stage 3: Image Sanitization (Safe RGB & Alpha Flattening)
    # -------------------------------------------------------------
    try:
        if pil_img.mode == "RGBA":
            # Flatten RGBA onto black background (standard background for brain MRI scans)
            bg = Image.new("RGB", pil_img.size, (0, 0, 0))
            bg.paste(pil_img, mask=pil_img.split()[3])
            sanitized_rgb = bg
        elif pil_img.mode in ("L", "1"):
            sanitized_rgb = pil_img.convert("RGB")
        elif pil_img.mode == "RGB":
            sanitized_rgb = pil_img.copy()
        elif pil_img.mode == "P":
            sanitized_rgb = pil_img.convert("RGBA")
            bg = Image.new("RGB", sanitized_rgb.size, (0, 0, 0))
            bg.paste(sanitized_rgb, mask=sanitized_rgb.split()[3])
            sanitized_rgb = bg
        else:
            sanitized_rgb = pil_img.convert("RGB")
    except Exception as conv_err:
        checks.append(ValidationCheck("Image Sanitization", False, ValidationStatus.REJECTED, str(conv_err)))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Color Mode Conversion Failed",
            message="Unable to safely convert image color channels for inference.",
            checks=checks,
            metrics=metrics
        )

    # -------------------------------------------------------------
    # Stage 4: Signal, Contrast & Uniformity Analysis
    # -------------------------------------------------------------
    np_arr = np.array(sanitized_rgb, dtype=np.float32)
    gray_arr = np.array(sanitized_rgb.convert("L"), dtype=np.float32)
    
    mean_intensity = float(np.mean(gray_arr))
    std_intensity = float(np.std(gray_arr))
    min_val = float(np.min(gray_arr))
    max_val = float(np.max(gray_arr))
    
    metrics["mean_intensity"] = round(mean_intensity, 2)
    metrics["std_intensity"] = round(std_intensity, 2)
    metrics["dynamic_range"] = round(max_val - min_val, 2)
    
    # Calculate dominant uniform pixel percentage
    int_gray = gray_arr.astype(np.uint8)
    _, counts = np.unique(int_gray, return_counts=True)
    max_freq_ratio = float(np.max(counts) / counts.sum())
    metrics["uniform_pixel_ratio"] = round(max_freq_ratio, 3)
    
    # Check for blank / solid black / solid white
    if std_intensity < 2.0 or (max_val - min_val) < 5.0:
        checks.append(ValidationCheck(
            "Signal & Contrast",
            False,
            ValidationStatus.REJECTED,
            f"Image is completely blank or uniform (Standard deviation: {std_intensity:.2f})."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Blank or Uniform Image",
            message="The uploaded image appears to be completely blank or uniform, with no detectable anatomical signal.",
            checks=checks,
            metrics=metrics
        )
        
    if max_freq_ratio > 0.98:
        checks.append(ValidationCheck(
            "Uniformity Screening",
            False,
            ValidationStatus.REJECTED,
            f"Over {max_freq_ratio*100:.1f}% of pixels are identical."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Extremely Uniform Image",
            message="Over 98% of the image contains an identical flat color with negligible structural content.",
            checks=checks,
            metrics=metrics
        )

    checks.append(ValidationCheck(
        "Signal & Dynamic Range",
        True,
        ValidationStatus.PASS,
        f"Valid visual contrast (Std: {std_intensity:.1f}, Dynamic range: {max_val - min_val:.0f})."
    ))

    # -------------------------------------------------------------
    # Stage 5: Conservative Brain / MRI Suitability Screening
    # -------------------------------------------------------------
    # Brain MRI scans are fundamentally grayscale structural sequences (monochromatic).
    # Natural photos, web graphics, and colorful illustrations have high chromatic saturation.
    r, g, b = np_arr[:, :, 0], np_arr[:, :, 1], np_arr[:, :, 2]
    chroma_diff = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    mean_chroma = float(np.mean(chroma_diff))
    metrics["chromatic_divergence"] = round(mean_chroma, 2)
    
    suitability_passed = True
    suitability_status = ValidationStatus.PASS
    suitability_msg = "Image exhibits visual properties consistent with grayscale neuroimaging slices."
    
    # Check 1: High chromatic divergence (vibrant natural photo, cartoon, landscape)
    if mean_chroma > 25.0:
        suitability_passed = False
        suitability_status = ValidationStatus.WARNING
        suitability_msg = (
            f"Prominent color saturation detected (chroma score: {mean_chroma:.1f}). "
            "Standard brain MRI scans are monochromatic. Prediction reliability cannot be guaranteed."
        )
    # Check 2: Pure inverted document / white-background document screenshot
    elif mean_intensity > 235.0 and std_intensity < 25.0:
        suitability_passed = False
        suitability_status = ValidationStatus.WARNING
        suitability_msg = "High average brightness and low variance resemble a text document or document screenshot."
        
    checks.append(ValidationCheck("Suitability Screening", suitability_passed, suitability_status, suitability_msg))

    # -------------------------------------------------------------
    # Consolidated Status Assessment
    # -------------------------------------------------------------
    overall_status = ValidationStatus.PASS
    warning_reasons = []
    for c in checks:
        if c.status == ValidationStatus.WARNING:
            overall_status = ValidationStatus.WARNING
            warning_reasons.append(c.message)
            
    if overall_status == ValidationStatus.PASS:
        headline = "Input Checks Passed"
        message = "The image passed all basic input and quality checks and is suitable for analysis by the research model."
    else:
        headline = "Suitability Warning"
        details_txt = " ".join(warning_reasons)
        message = f"The image can be processed, but suitability could not be confidently established: {details_txt}"
        
    return ValidationResult(
        status=overall_status,
        headline=headline,
        message=message,
        checks=checks,
        metrics=metrics,
        sanitized_image=sanitized_rgb
    )


def compute_uncertainty_metrics(probabilities: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate mathematical uncertainty metrics from the 4-class softmax probability distribution.
    Uses normalized Shannon entropy and Top-1 vs Top-2 prediction margin.
    
    Important: This is a mathematical measure of probability dispersion across classes,
    NOT a clinically validated diagnostic uncertainty score.
    """
    probs = np.array(list(probabilities.values()), dtype=np.float64)
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / np.sum(probs)
    
    # Normalized Shannon Entropy: H = -sum(p * log4(p))
    # Base 4 normalizes entropy to exactly [0.0, 1.0] for 4 classes
    entropy = -np.sum(probs * (np.log(probs) / np.log(4.0)))
    entropy = float(np.clip(entropy, 0.0, 1.0))
    
    # Margin of confidence: difference between highest and second highest probability
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = float(top1 - top2)
    
    if entropy < 0.35 and margin > 0.60:
        level = "Low"
        badge_class = "badge-chip-success"
        description = "Prediction is sharply focused on a single stage with high probability margin."
    elif entropy < 0.70 and margin > 0.20:
        level = "Moderate"
        badge_class = "badge-chip-warning"
        description = "Probability mass is somewhat distributed across adjacent clinical stages."
    else:
        level = "High"
        badge_class = "badge-chip-danger"
        description = "Significant probability dispersion across multiple stages. Model exhibits substantial uncertainty."
        
    return {
        "normalized_entropy": round(entropy, 3),
        "prediction_margin": round(margin, 3),
        "uncertainty_level": level,
        "badge_class": badge_class,
        "description": description,
        "disclaimer": "Derived from mathematical Shannon entropy of the softmax output. Not a clinically validated uncertainty metric."
    }
