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

    # Aspect ratio check (axial brain MRIs are near-square)
    aspect_check_passed = True
    aspect_status = ValidationStatus.PASS
    aspect_msg = f"Aspect ratio ({aspect_ratio:.2f}:1) is standard."
    
    if aspect_ratio > 1.8 or aspect_ratio < 0.55:
        aspect_check_passed = False
        aspect_status = ValidationStatus.REJECTED
        aspect_msg = f"Unsuitable aspect ratio ({aspect_ratio:.2f}:1). Axial brain MRI scans have near-square proportions."
        checks.append(ValidationCheck("Aspect Ratio", False, aspect_status, aspect_msg))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
            checks=checks,
            metrics=metrics
        )
    elif aspect_ratio > 1.45 or aspect_ratio < 0.70:
        aspect_check_passed = False
        aspect_status = ValidationStatus.WARNING
        aspect_msg = f"Unusual aspect ratio ({aspect_ratio:.2f}:1). Standard brain MRI scans typically have near-square proportions."
        
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
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
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
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
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
    # Stage 5: Strict Brain MRI Domain Verification (Multi-Feature Screener)
    # -------------------------------------------------------------
    # The research platform is trained exclusively on axial brain MRI scans (T1-weighted).
    # Inputs outside this domain (natural photos, animals, human faces, digital art,
    # screenshots, chest X-rays/CT, spine, abdomen, knee/shoulder scans) must be strictly REJECTED.
    
    # 1. Chromatic Saturation Check
    # Brain MRI scans are strictly monochromatic grayscale.
    r, g, b = np_arr[:, :, 0], np_arr[:, :, 1], np_arr[:, :, 2]
    chroma_diff = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    mean_chroma = float(np.mean(chroma_diff))
    metrics["chromatic_divergence"] = round(mean_chroma, 2)
    
    if mean_chroma > 8.0:
        checks.append(ValidationCheck(
            "Domain Suitability (Color)",
            False,
            ValidationStatus.REJECTED,
            f"Color saturation detected (chromatic divergence: {mean_chroma:.1f}). Brain MRI scans are monochromatic grayscale neuroimaging sequences."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
            checks=checks,
            metrics=metrics,
            sanitized_image=sanitized_rgb
        )

    # 2. Perimeter Background Darkness Check (Outer 5% Margin)
    # In axial brain MRI, the head is centered and surrounded by dark scanner air background.
    bw = max(2, int(width * 0.05))
    bh = max(2, int(height * 0.05))
    top = gray_arr[:bh, :]
    bottom = gray_arr[-bh:, :]
    left = gray_arr[:, :bw]
    right = gray_arr[:, -bw:]
    border_pixels = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])
    border_mean = float(np.mean(border_pixels))
    metrics["border_mean_intensity"] = round(border_mean, 2)
    
    if border_mean > 30.0:
        checks.append(ValidationCheck(
            "Domain Suitability (Perimeter)",
            False,
            ValidationStatus.REJECTED,
            f"Non-dark perimeter detected (border mean: {border_mean:.1f}). Axial brain MRI scans must feature a dark scanner background surrounding the cranium."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
            checks=checks,
            metrics=metrics,
            sanitized_image=sanitized_rgb
        )

    # 3. Corner Background Darkness Check (Outer 8% Corners)
    # In axial cranial imaging, all 4 corners lie outside the cranial oval in empty scanner space.
    cw = max(2, int(width * 0.08))
    ch = max(2, int(height * 0.08))
    corners = np.concatenate([
        gray_arr[:ch, :cw].flatten(),
        gray_arr[:ch, -cw:].flatten(),
        gray_arr[-ch:, :cw].flatten(),
        gray_arr[-ch:, -cw:].flatten()
    ])
    corner_mean = float(np.mean(corners))
    metrics["corner_mean_intensity"] = round(corner_mean, 2)
    
    if corner_mean > 30.0:
        checks.append(ValidationCheck(
            "Domain Suitability (Corners)",
            False,
            ValidationStatus.REJECTED,
            f"Non-dark corners detected (corner mean: {corner_mean:.1f}). In axial brain MRI, frame corners are empty scanner background."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
            checks=checks,
            metrics=metrics,
            sanitized_image=sanitized_rgb
        )

    # 4. Anatomical Tissue Coverage (Foreground Fraction of Pixels > 20)
    # True axial brain slices cover 35%-60% of the field of view.
    fg_fraction = float(np.mean(gray_arr > 20.0))
    metrics["tissue_coverage_fraction"] = round(fg_fraction, 3)
    
    if fg_fraction < 0.28:
        checks.append(ValidationCheck(
            "Domain Suitability (Tissue Coverage)",
            False,
            ValidationStatus.REJECTED,
            f"Insufficient anatomical tissue coverage ({fg_fraction*100:.1f}%). Axial brain slices typically cover 35%-60% of the field of view."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
            checks=checks,
            metrics=metrics,
            sanitized_image=sanitized_rgb
        )
    if fg_fraction > 0.68:
        checks.append(ValidationCheck(
            "Domain Suitability (Tissue Coverage)",
            False,
            ValidationStatus.REJECTED,
            f"Excessive tissue coverage ({fg_fraction*100:.1f}%). Subject fills entire frame without characteristic cranial air boundary."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
            checks=checks,
            metrics=metrics,
            sanitized_image=sanitized_rgb
        )

    # 5. Central Brain Parenchyma Signal & Contrast
    # Axial brain slices contain dense brain parenchyma in the central 50%.
    center = gray_arr[int(height * 0.25):int(height * 0.75), int(width * 0.25):int(width * 0.75)]
    center_mean = float(np.mean(center))
    metrics["center_mean_intensity"] = round(center_mean, 2)
    cranial_contrast = center_mean - border_mean
    metrics["cranial_contrast"] = round(cranial_contrast, 2)
    
    if center_mean < 45.0:
        checks.append(ValidationCheck(
            "Domain Suitability (Parenchyma)",
            False,
            ValidationStatus.REJECTED,
            f"Low central parenchyma signal ({center_mean:.1f}). Axial brain neuroimaging displays solid cerebral tissue in the center."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
            checks=checks,
            metrics=metrics,
            sanitized_image=sanitized_rgb
        )
    if cranial_contrast < 25.0:
        checks.append(ValidationCheck(
            "Domain Suitability (Contrast)",
            False,
            ValidationStatus.REJECTED,
            f"Insufficient cranial-to-background contrast ({cranial_contrast:.1f}). Expected distinct boundary between brain tissue and scanner background."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
            checks=checks,
            metrics=metrics,
            sanitized_image=sanitized_rgb
        )

    # 6. Cranial Anatomical Geometry (Aspect Ratio of Central Tissue)
    mid_row = gray_arr[height // 2, :]
    mid_col = gray_arr[:, width // 2]
    row_nz = np.where(mid_row > 20.0)[0]
    col_nz = np.where(mid_col > 20.0)[0]
    if len(row_nz) > 0 and len(col_nz) > 0:
        skull_w = row_nz[-1] - row_nz[0]
        skull_h = col_nz[-1] - col_nz[0]
        skull_ratio = skull_w / max(skull_h, 1)
        metrics["cranial_aspect_ratio"] = round(skull_ratio, 2)
        if skull_ratio < 0.68 or skull_ratio > 1.42:
            checks.append(ValidationCheck(
                "Domain Suitability (Geometry)",
                False,
                ValidationStatus.REJECTED,
                f"Abnormal anatomical geometry (cranial width/height ratio: {skull_ratio:.2f}). Axial brain cross-sections are approximately elliptical/round."
            ))
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                headline="Input rejected",
                message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
                checks=checks,
                metrics=metrics,
                sanitized_image=sanitized_rgb
            )

    # 7. Bilateral Hemispheric Symmetry Check
    half_w = width // 2
    left_side = gray_arr[:, :half_w]
    right_side = np.fliplr(gray_arr[:, -half_w:])
    asym = float(np.mean(np.abs(left_side - right_side)) / max(mean_intensity, 1.0))
    metrics["hemispheric_asymmetry"] = round(asym, 3)
    if asym > 0.65:
        checks.append(ValidationCheck(
            "Domain Suitability (Symmetry)",
            False,
            ValidationStatus.REJECTED,
            f"High structural asymmetry ({asym:.2f}). Axial neuroimaging exhibits bilateral hemispheric symmetry across the sagittal midline."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
            checks=checks,
            metrics=metrics,
            sanitized_image=sanitized_rgb
        )

    # 8. Vertical Perimeter Continuity (Spine Check)
    # Sagittal spine scans have vertebral tissue continuous with top and bottom edges.
    top_mid = float(np.mean(gray_arr[:bh, int(width * 0.35):int(width * 0.65)]))
    bot_mid = float(np.mean(gray_arr[-bh:, int(width * 0.35):int(width * 0.65)]))
    if top_mid > 30.0 and bot_mid > 30.0:
        checks.append(ValidationCheck(
            "Domain Suitability (Spine Exclusion)",
            False,
            ValidationStatus.REJECTED,
            "Continuous vertical tissue intersecting top and bottom boundaries (characteristic of spine MRI, not axial brain MRI)."
        ))
        return ValidationResult(
            status=ValidationStatus.REJECTED,
            headline="Input rejected",
            message="This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.",
            checks=checks,
            metrics=metrics,
            sanitized_image=sanitized_rgb
        )

    checks.append(ValidationCheck(
        "Brain MRI Domain Verification",
        True,
        ValidationStatus.PASS,
        "Image conforms to studied axial brain MRI domain characteristics (monochromatic, dark perimeter, centered cranial parenchyma)."
    ))

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
        headline = "Brain MRI Input Verified"
        message = "The image conforms to the studied axial brain MRI domain and passed all quality screening checks."
    else:
        headline = "Suitability Warning"
        details_txt = " ".join(warning_reasons)
        message = f"The image can be processed, but quality notes were flagged: {details_txt}"
        
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
