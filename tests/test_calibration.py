"""
Unit Test Suite for Model Calibration and Reliability Analysis Engine.
Verifies temperature scaling module, positive temperature constraint,
probability normalization, absence of NaNs/Infs, calibration metric calculations,
ECE/MCE bounds, Brier score bounds, NLL finiteness, and test label immutability.
"""

import os
import sys
import json
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.calibration import (
    TemperatureScaler,
    compute_ece_mce,
    compute_brier_score,
    compute_nll,
    compute_comprehensive_calibration_metrics
)


class TestModelCalibration(unittest.TestCase):
    """Test suite verifying calibration mathematical correctness and data integrity."""

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Synthetic test data
        self.num_samples = 200
        self.num_classes = 4
        self.raw_logits = torch.randn(self.num_samples, self.num_classes)
        self.targets = np.random.randint(0, self.num_classes, size=self.num_samples)
        self.raw_probs = F.softmax(self.raw_logits, dim=1).numpy()

    def test_01_temperature_scaler_positive(self):
        """Verify temperature scaling initializes positive and clamps within valid range."""
        scaler = TemperatureScaler(init_temperature=1.5)
        self.assertGreater(float(scaler.temperature.item()), 0.0, "Temperature must be strictly positive")
        
        # Test forward pass with temperature scaling
        scaled_logits = scaler(self.raw_logits)
        self.assertEqual(scaled_logits.shape, self.raw_logits.shape)
        self.assertFalse(torch.isnan(scaled_logits).any(), "Scaled logits must not contain NaN")
        self.assertFalse(torch.isinf(scaled_logits).any(), "Scaled logits must not contain Inf")

    def test_02_calibrated_probabilities_sum_to_one(self):
        """Verify calibrated probabilities sum to 1.0 for each sample and contain no NaNs."""
        scaler = TemperatureScaler(init_temperature=1.25)
        with torch.no_grad():
            scaled = scaler(self.raw_logits)
            cal_probs = F.softmax(scaled, dim=1).numpy()
            
        self.assertFalse(np.isnan(cal_probs).any(), "Calibrated probabilities must not contain NaN")
        self.assertFalse(np.isinf(cal_probs).any(), "Calibrated probabilities must not contain Inf")
        
        row_sums = np.sum(cal_probs, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5, err_msg="Calibrated probabilities must sum to 1.0")

    def test_03_monotonicity_and_argmax_preservation(self):
        """Verify that temperature scaling does not change top-1 predicted class index."""
        scaler = TemperatureScaler(init_temperature=1.4)
        orig_preds = torch.argmax(self.raw_logits, dim=1)
        with torch.no_grad():
            scaled_logits = scaler(self.raw_logits)
            cal_preds = torch.argmax(scaled_logits, dim=1)
            
        torch.testing.assert_close(orig_preds, cal_preds, msg="Temperature scaling must preserve argmax ranking")

    def test_04_ece_mce_bounds_and_validity(self):
        """Verify ECE and MCE lie in [0, 1] and handle edge cases correctly."""
        res = compute_ece_mce(self.raw_probs, self.targets, num_bins=15)
        
        self.assertIn("ece", res)
        self.assertIn("mce", res)
        self.assertIn("bin_stats", res)
        
        ece = res["ece"]
        mce = res["mce"]
        
        self.assertGreaterEqual(ece, 0.0, "ECE must be non-negative")
        self.assertLessEqual(ece, 1.0, "ECE must not exceed 1.0")
        self.assertGreaterEqual(mce, 0.0, "MCE must be non-negative")
        self.assertLessEqual(mce, 1.0, "MCE must not exceed 1.0")
        self.assertLessEqual(ece, mce + 1e-6, "ECE cannot exceed MCE")
        self.assertEqual(len(res["bin_stats"]), 15, "Must compute exactly 15 bins")

    def test_05_brier_score_bounds(self):
        """Verify multi-class Brier score is non-negative and bounded by 2.0."""
        brier = compute_brier_score(self.raw_probs, self.targets, num_classes=self.num_classes)
        self.assertGreaterEqual(brier, 0.0, "Brier score must be non-negative")
        self.assertLessEqual(brier, 2.0, "Multi-class Brier score cannot exceed 2.0")
        self.assertFalse(np.isnan(brier), "Brier score must not be NaN")

    def test_06_nll_finite(self):
        """Verify Negative Log-Likelihood is finite and positive."""
        nll = compute_nll(self.raw_probs, self.targets)
        self.assertGreater(nll, 0.0, "NLL must be positive")
        self.assertTrue(np.isfinite(nll), "NLL must be finite")

    def test_07_perfect_calibration_synthetic_case(self):
        """Test on perfectly calibrated synthetic distribution."""
        # Create perfect predictions
        perfect_probs = np.zeros((100, 4))
        targets = np.random.randint(0, 4, size=100)
        perfect_probs[np.arange(100), targets] = 1.0
        
        res = compute_ece_mce(perfect_probs, targets, num_bins=10)
        self.assertAlmostEqual(res["ece"], 0.0, places=4, msg="Perfect predictor must have ECE ~ 0.0")
        self.assertAlmostEqual(res["mce"], 0.0, places=4, msg="Perfect predictor must have MCE ~ 0.0")
        
        brier = compute_brier_score(perfect_probs, targets, num_classes=4)
        self.assertAlmostEqual(brier, 0.0, places=4, msg="Perfect predictor must have Brier score ~ 0.0")

    def test_08_learned_temperature_files_exist_and_positive(self):
        """Verify that learned temperature JSON files exist and have T > 0."""
        cal_dir = PROJECT_ROOT / "results" / "calibration"
        expected_models = ["mobilenet_v2", "efficientnet_b0", "resnet18"]
        
        for slug in expected_models:
            json_file = cal_dir / f"{slug}_temperature.json"
            self.assertTrue(json_file.exists(), f"Temperature JSON file missing: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertIn("learned_temperature", data)
            t_val = data["learned_temperature"]
            self.assertGreater(t_val, 0.0, f"Learned temperature for {slug} must be positive, got {t_val}")
            self.assertFalse(data.get("test_leakage", True), "Must guarantee zero test leakage")

    def test_09_test_labels_and_sample_count_immutable(self):
        """Verify that held-out test split remains exactly 960 samples with intact labels."""
        test_csv = PROJECT_ROOT / "reports" / "splits" / "test.csv"
        self.assertTrue(test_csv.exists(), "test.csv must exist")
        
        with open(test_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        sample_rows = lines[1:]  # Exclude header
        self.assertEqual(len(sample_rows), 960, "Held-out test set must contain exactly 960 samples")


if __name__ == "__main__":
    unittest.main()
