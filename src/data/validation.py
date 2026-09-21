"""
Input validation and dataset integrity checking module.
Defines canonical classes, display names, image validation, and dataset structure verification.
"""

import os
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np

# Canonical 4-stage Alzheimer's Disease classes
CLASSES = [
    'Non_Demented',
    'Very_Mild_Demented',
    'Mild_Demented',
    'Moderate_Demented'
]

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# Clean display names for UI and reporting
CLASS_DISPLAY_NAMES = {
    'Non_Demented': 'Non-Demented (Control)',
    'Very_Mild_Demented': 'Very Mild Demented (Early Stage)',
    'Mild_Demented': 'Mild Demented (Intermediate Stage)',
    'Moderate_Demented': 'Moderate Demented (Advanced Stage)'
}

# Image validation constraints
MIN_IMAGE_DIM = 32
MAX_IMAGE_DIM = 4096
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def validate_image_file(file_input: Any) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validate an uploaded file or file path as a valid, non-corrupted Brain MRI image.
    
    Args:
        file_input: File path (str), BytesIO stream, or uploaded file object.
        
    Returns:
        Tuple of (is_valid: bool, error_message: str, image: Optional[Image.Image])
    """
    if file_input is None:
        return False, "No file provided.", None
    
    try:
        if isinstance(file_input, str):
            if not os.path.exists(file_input):
                return False, f"File does not exist: {file_input}", None
            ext = os.path.splitext(file_input)[1].lower()
            if ext not in VALID_EXTENSIONS:
                return False, f"Unsupported file extension '{ext}'. Allowed: {VALID_EXTENSIONS}", None
            img = Image.open(file_input)
        else:
            # Handles BytesIO or UploadedFile objects (Streamlit)
            img = Image.open(file_input)
        
        # Verify image integrity
        img.verify()
        
        # Re-open after verify (PIL requirement)
        if isinstance(file_input, str):
            img = Image.open(file_input)
        else:
            file_input.seek(0)
            img = Image.open(file_input)
            
        width, height = img.size
        
        if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
            return False, f"Image resolution ({width}x{height}) is too low (minimum is {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}).", None
            
        if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
            return False, f"Image resolution ({width}x{height}) exceeds maximum limit ({MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}).", None
            
        # Convert to numpy array to check for blank/solid color images
        img_np = np.array(img.convert('L'))
        if img_np.std() < 1e-3:
            return False, "Image appears completely blank or uniform (insufficient variance for MRI analysis).", None
            
        return True, "Valid Brain MRI image.", img.convert('RGB')
        
    except Exception as e:
        return False, f"Failed to decode image: {str(e)}", None


def validate_dataset_directory(dataset_dir: str) -> Dict[str, Any]:
    """
    Check the dataset folder to verify class subdirectories and count samples.
    
    Args:
        dataset_dir: Path to data/dataset directory.
        
    Returns:
        Dictionary containing validation status, class sample counts, total samples, and messages.
    """
    result = {
        "exists": False,
        "is_ready": False,
        "class_counts": {cls_name: 0 for cls_name in CLASSES},
        "total_samples": 0,
        "missing_folders": [],
        "message": ""
    }
    
    if not os.path.exists(dataset_dir):
        result["message"] = f"Dataset directory not found at: {dataset_dir}"
        return result
        
    result["exists"] = True
    missing = []
    total = 0
    
    for cls_name in CLASSES:
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.isdir(cls_dir):
            missing.append(cls_name)
        else:
            files = [f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
            count = len(files)
            result["class_counts"][cls_name] = count
            total += count
            
    result["total_samples"] = total
    result["missing_folders"] = missing
    
    if missing:
        result["message"] = f"Missing class subdirectories: {', '.join(missing)}"
        result["is_ready"] = False
    elif total == 0:
        result["message"] = "Class subdirectories exist, but no image files were found in them."
        result["is_ready"] = False
    else:
        result["is_ready"] = True
        result["message"] = f"Dataset valid and ready. Total samples: {total} across {len(CLASSES)} stages."
        
    return result
