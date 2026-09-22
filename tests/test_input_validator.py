"""
Comprehensive Test Suite for Robust Input Validation & Safe Inference Gate.
Tests file integrity, decoding, resolutions, color modes, blank images,
corrupted data, extreme aspect ratios, and suitability screening.
"""

import sys
import unittest
import io
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.input_validator import (
    validate_input_image,
    compute_uncertainty_metrics,
    ValidationStatus
)
from app.utils.data_loader import get_sample_test_images


class TestInputValidator(unittest.TestCase):
    """Test suite for input_validator.py."""

    @classmethod
    def setUpClass(cls):
        # Obtain a real dataset MRI scan
        sample_dict = get_sample_test_images()
        cls.real_mri_path = sample_dict["Non-Demented"][0]["path"]
        assert Path(cls.real_mri_path).exists(), f"Real MRI missing: {cls.real_mri_path}"

    def test_case_a_valid_dataset_mri(self):
        """Test Case A: Valid dataset MRI image should PASS."""
        res = validate_input_image(self.real_mri_path)
        self.assertEqual(res.status, ValidationStatus.PASS)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.can_run_inference)
        self.assertIsNotNone(res.sanitized_image)

    def test_case_b_grayscale_image(self):
        """Test Case B: Grayscale mode MRI should PASS."""
        img = Image.open(self.real_mri_path).convert("L")
        res = validate_input_image(img)
        self.assertEqual(res.status, ValidationStatus.PASS)
        self.assertEqual(res.sanitized_image.mode, "RGB")

    def test_case_c_rgb_image(self):
        """Test Case C: 3-channel RGB image should PASS."""
        img = Image.open(self.real_mri_path).convert("RGB")
        res = validate_input_image(img)
        self.assertEqual(res.status, ValidationStatus.PASS)
        self.assertEqual(res.sanitized_image.mode, "RGB")

    def test_case_d_rgba_image_flattening(self):
        """Test Case D: RGBA image should be converted/flattened onto black background safely."""
        img = Image.open(self.real_mri_path).convert("RGBA")
        res = validate_input_image(img)
        self.assertEqual(res.status, ValidationStatus.PASS)
        self.assertEqual(res.sanitized_image.mode, "RGB")
        self.assertEqual(res.sanitized_image.size, img.size)

    def test_case_e_different_resolutions(self):
        """Test Case E: Different resolutions (128x128, 256x256, 512x512, 1024x1024) handled safely."""
        base_img = Image.open(self.real_mri_path)
        for size in [(128, 128), (256, 256), (512, 512), (1024, 1024)]:
            resized = base_img.resize(size, Image.Resampling.BILINEAR)
            res = validate_input_image(resized)
            self.assertEqual(res.status, ValidationStatus.PASS, f"Failed at size {size}")
            self.assertEqual(res.metrics["width"], size[0])
            self.assertEqual(res.metrics["height"], size[1])

    def test_case_f_corrupted_image(self):
        """Test Case F: Corrupted image bytes should be REJECTED."""
        corrupt_bytes = b"GIF89a\x00\x00NOT_AN_IMAGE_DATA_CORRUPT"
        res = validate_input_image(corrupt_bytes)
        self.assertEqual(res.status, ValidationStatus.REJECTED)
        self.assertFalse(res.can_run_inference)

    def test_case_g_unsupported_file_format(self):
        """Test Case G: Unsupported file format (e.g. BMP/GIF or text) should be REJECTED."""
        # Create a BMP in memory
        img = Image.new("RGB", (100, 100), color="blue")
        bio = io.BytesIO()
        img.save(bio, format="BMP")
        res = validate_input_image(bio.getvalue(), filename="test.bmp")
        self.assertEqual(res.status, ValidationStatus.REJECTED)
        self.assertIn("BMP", res.message)

    def test_case_h_blank_images(self):
        """Test Case H: Pure black or pure white image should be REJECTED."""
        # Pure black
        black_img = Image.new("RGB", (128, 128), color=(0, 0, 0))
        res_black = validate_input_image(black_img)
        self.assertEqual(res_black.status, ValidationStatus.REJECTED)
        self.assertFalse(res_black.can_run_inference)

        # Pure white
        white_img = Image.new("RGB", (128, 128), color=(255, 255, 255))
        res_white = validate_input_image(white_img)
        self.assertEqual(res_white.status, ValidationStatus.REJECTED)
        self.assertFalse(res_white.can_run_inference)

    def test_case_i_near_uniform_image(self):
        """Test Case I: Near-uniform flat color image should be REJECTED."""
        # Solid flat gray with minor noise
        gray_arr = np.full((128, 128), 120, dtype=np.uint8)
        gray_arr[0, 0] = 121
        flat_img = Image.fromarray(gray_arr, mode="L")
        res = validate_input_image(flat_img)
        self.assertEqual(res.status, ValidationStatus.REJECTED)

    def test_case_j_extreme_aspect_ratio(self):
        """Test Case J: Extreme aspect ratio should be WARNING or REJECTED."""
        # Extreme strip (500x50 -> 10:1 ratio)
        strip_img = Image.new("RGB", (500, 50), color=(50, 50, 50))
        # add some non-uniform pixels to avoid blank reject
        for x in range(500):
            strip_img.putpixel((x, 25), (150, 150, 150))
        res = validate_input_image(strip_img)
        self.assertEqual(res.status, ValidationStatus.REJECTED)

        # Moderate deviation (300x120 -> 2.5:1 ratio) with varied pixel texture
        mod_arr = np.random.randint(40, 200, (120, 300, 3), dtype=np.uint8)
        mod_img = Image.fromarray(mod_arr)
        res_mod = validate_input_image(mod_img)
        self.assertEqual(res_mod.status, ValidationStatus.WARNING)

    def test_case_k_unrelated_vibrant_photo(self):
        """Test Case K: Vibrant chromatic natural image should receive WARNING."""
        # High saturation color image (red, green, blue bands)
        rgb_arr = np.zeros((128, 128, 3), dtype=np.uint8)
        rgb_arr[:40, :, 0] = 255  # Red
        rgb_arr[40:80, :, 1] = 255  # Green
        rgb_arr[80:, :, 2] = 255  # Blue
        vibrant_img = Image.fromarray(rgb_arr)
        res = validate_input_image(vibrant_img)
        self.assertEqual(res.status, ValidationStatus.WARNING)
        self.assertIn("color saturation", res.message.lower())

    def test_uncertainty_metrics(self):
        """Test uncertainty metrics calculation from probability distributions."""
        # Low uncertainty (peaked)
        peaked_probs = {"Non-Demented": 0.98, "Very Mild Demented": 0.01, "Mild Demented": 0.005, "Moderate Demented": 0.005}
        u_peaked = compute_uncertainty_metrics(peaked_probs)
        self.assertEqual(u_peaked["uncertainty_level"], "Low")
        self.assertLess(u_peaked["normalized_entropy"], 0.35)

        # High uncertainty (dispersed across stages)
        flat_probs = {"Non-Demented": 0.28, "Very Mild Demented": 0.26, "Mild Demented": 0.24, "Moderate Demented": 0.22}
        u_flat = compute_uncertainty_metrics(flat_probs)
        self.assertEqual(u_flat["uncertainty_level"], "High")
        self.assertGreater(u_flat["normalized_entropy"], 0.70)


if __name__ == "__main__":
    unittest.main()
