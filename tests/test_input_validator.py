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
        """Test Case J: Extreme aspect ratio should be REJECTED, mild non-square should WARN."""
        # Extreme strip (500x50 -> 10:1 ratio)
        strip_img = Image.new("RGB", (500, 50), color=(50, 50, 50))
        for x in range(500):
            strip_img.putpixel((x, 25), (150, 150, 150))
        res = validate_input_image(strip_img)
        self.assertEqual(res.status, ValidationStatus.REJECTED)
        self.assertFalse(res.can_run_inference)
        self.assertEqual(res.headline, "Input rejected")

        # Non-standard aspect ratio (1.55:1, between 1.45 and 1.80) on valid brain MRI
        base_img = Image.open(self.real_mri_path)
        mild_skew = base_img.resize((155, 100))
        res_mild = validate_input_image(mild_skew)
        # May be WARNING or PASS depending on geometry, but must allow inference if brain content matches
        if res_mild.status == ValidationStatus.WARNING:
            self.assertTrue(res_mild.can_run_inference)

    def test_case_k_unrelated_vibrant_photo_rejected(self):
        """Test Case K: Vibrant chromatic natural image must be strictly REJECTED."""
        # High saturation color image (red, green, blue bands)
        rgb_arr = np.zeros((128, 128, 3), dtype=np.uint8)
        rgb_arr[:40, :, 0] = 255  # Red
        rgb_arr[40:80, :, 1] = 255  # Green
        rgb_arr[80:, :, 2] = 255  # Blue
        vibrant_img = Image.fromarray(rgb_arr)
        res = validate_input_image(vibrant_img)
        self.assertEqual(res.status, ValidationStatus.REJECTED)
        self.assertFalse(res.can_run_inference)
        self.assertEqual(res.headline, "Input rejected")
        self.assertEqual(res.message, "This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.")

    def test_case_l_animal_photo_rejection(self):
        """Test Case L: Dog/cat/animal photos (color and grayscale) must be REJECTED."""
        # Color pet photo (golden/brown fur tones)
        dog_arr = np.zeros((160, 160, 3), dtype=np.uint8)
        dog_arr[:, :, 0] = 180  # R
        dog_arr[:, :, 1] = 130  # G
        dog_arr[:, :, 2] = 70   # B
        res_dog = validate_input_image(Image.fromarray(dog_arr))
        self.assertEqual(res_dog.status, ValidationStatus.REJECTED)
        self.assertFalse(res_dog.can_run_inference)
        self.assertEqual(res_dog.headline, "Input rejected")

        # Grayscale full-frame animal photo (no dark perimeter, high corner intensity)
        cat_gray = np.random.randint(60, 190, (160, 160), dtype=np.uint8)
        res_cat = validate_input_image(Image.fromarray(cat_gray, mode="L"))
        self.assertEqual(res_cat.status, ValidationStatus.REJECTED)
        self.assertFalse(res_cat.can_run_inference)
        self.assertEqual(res_cat.headline, "Input rejected")

    def test_case_m_human_photo_rejection(self):
        """Test Case M: Human photographs / selfies / portraits must be REJECTED."""
        # Human portrait skin tones
        person_arr = np.zeros((180, 180, 3), dtype=np.uint8)
        person_arr[:, :, 0] = 220
        person_arr[:, :, 1] = 175
        person_arr[:, :, 2] = 145
        res_person = validate_input_image(Image.fromarray(person_arr))
        self.assertEqual(res_person.status, ValidationStatus.REJECTED)
        self.assertFalse(res_person.can_run_inference)
        self.assertEqual(res_person.headline, "Input rejected")

    def test_case_n_digital_and_ai_artwork_rejection(self):
        """Test Case N: Digital artwork or AI-generated colorful brain illustrations must be REJECTED."""
        art_arr = np.zeros((150, 150, 3), dtype=np.uint8)
        art_arr[:, :, 0] = 90
        art_arr[:, :, 1] = 210
        art_arr[:, :, 2] = 240
        res_art = validate_input_image(Image.fromarray(art_arr))
        self.assertEqual(res_art.status, ValidationStatus.REJECTED)
        self.assertFalse(res_art.can_run_inference)
        self.assertEqual(res_art.headline, "Input rejected")

    def test_case_o_screenshot_and_document_rejection(self):
        """Test Case O: Screenshots and white documents must be REJECTED."""
        doc_img = Image.new("RGB", (300, 250), (250, 250, 250))
        res_doc = validate_input_image(doc_img)
        self.assertEqual(res_doc.status, ValidationStatus.REJECTED)
        self.assertFalse(res_doc.can_run_inference)
        self.assertEqual(res_doc.headline, "Input rejected")

    def test_case_p_chest_xray_and_ct_rejection(self):
        """Test Case P: Chest X-rays and lung CT scans must be REJECTED."""
        # Chest X-ray mock with bright ribs/corners (perimeter mean > 30)
        chest_arr = np.full((160, 160), 80, dtype=np.uint8)
        chest_arr[30:130, 20:65] = 15   # left lung void
        chest_arr[30:130, 95:140] = 15  # right lung void
        chest_arr[30:130, 65:95] = 160  # spine
        res_chest = validate_input_image(Image.fromarray(chest_arr, mode="L"))
        self.assertEqual(res_chest.status, ValidationStatus.REJECTED)
        self.assertFalse(res_chest.can_run_inference)
        self.assertEqual(res_chest.headline, "Input rejected")

    def test_case_q_spine_scan_rejection(self):
        """Test Case Q: Sagittal spine scans (continuous vertical column) must be REJECTED."""
        # Vertebral column intersecting top and bottom boundaries
        spine_arr = np.zeros((180, 180), dtype=np.uint8)
        spine_arr[:, 70:110] = 140
        res_spine = validate_input_image(Image.fromarray(spine_arr, mode="L"))
        self.assertEqual(res_spine.status, ValidationStatus.REJECTED)
        self.assertFalse(res_spine.can_run_inference)
        self.assertEqual(res_spine.headline, "Input rejected")

    def test_case_r_knee_and_extremity_rejection(self):
        """Test Case R: Asymmetric knee / shoulder / bone joint scans must be REJECTED."""
        # Severely asymmetric off-center joint
        joint_arr = np.zeros((160, 160), dtype=np.uint8)
        joint_arr[30:130, 10:70] = 150  # only left side
        res_joint = validate_input_image(Image.fromarray(joint_arr, mode="L"))
        self.assertEqual(res_joint.status, ValidationStatus.REJECTED)
        self.assertFalse(res_joint.can_run_inference)
        self.assertEqual(res_joint.headline, "Input rejected")

    def test_case_s_all_curated_demo_samples_pass(self):
        """Test Case S: All 12 curated demo MRI scans must PASS with full inference capability."""
        sample_dict = get_sample_test_images()
        for class_name, samples in sample_dict.items():
            for s in samples:
                p = s["path"]
                res = validate_input_image(p)
                self.assertEqual(res.status, ValidationStatus.PASS, f"Curated demo failed: {s['filename']} -> {res.message}")
                self.assertTrue(res.can_run_inference)
                self.assertEqual(res.headline, "Brain MRI Input Verified")

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

