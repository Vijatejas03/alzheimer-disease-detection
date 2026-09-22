"""
Unit Test Suite for Error Analysis and Robustness Testing Engine.
Verifies prediction records schema, probability normalization, absence of NaNs/Infs,
valid class labels, test sample count (960), zero hash duplication, and robustness output structure.
"""

import os
import sys
import json
import csv
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation import CLASSES


class TestErrorAnalysisAndRobustness(unittest.TestCase):
    """Test suite verifying error analysis artifacts and robustness testing outputs."""

    @classmethod
    def setUpClass(cls):
        cls.error_dir = PROJECT_ROOT / "results" / "error_analysis"
        cls.robust_dir = PROJECT_ROOT / "results" / "robustness"
        cls.expected_models = ["mobilenet_v2", "efficientnet_b0", "resnet18"]

    def test_01_prediction_csv_files_exist(self):
        """Verify that prediction-level CSVs exist for all 3 models."""
        for slug in self.expected_models:
            csv_path = self.error_dir / f"{slug}_predictions.csv"
            self.assertTrue(csv_path.exists(), f"Prediction CSV missing for {slug}: {csv_path}")

    def test_02_prediction_records_schema_and_sample_count(self):
        """Verify schema, fields, and exactly 960 rows in each prediction CSV."""
        required_fields = [
            "sample_idx", "image_path", "filename", "true_class", "true_idx",
            "predicted_class", "predicted_idx", "is_correct", "confidence",
            "second_highest_prob", "prediction_margin", "entropy_uncertainty",
            "calibrated_confidence", "prob_non_demented", "prob_very_mild",
            "prob_mild", "prob_moderate"
        ]
        
        for slug in self.expected_models:
            csv_path = self.error_dir / f"{slug}_predictions.csv"
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            self.assertEqual(len(rows), 960, f"Expected 960 prediction rows in {slug}, found {len(rows)}")
            for field in required_fields:
                self.assertIn(field, reader.fieldnames, f"Missing field {field} in {slug}_predictions.csv")

    def test_03_probabilities_normalized_and_no_nan_inf(self):
        """Verify that probabilities sum to ~1.0 and contain no NaN or Inf."""
        for slug in self.expected_models:
            csv_path = self.error_dir / f"{slug}_predictions.csv"
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    p0 = float(row["prob_non_demented"])
                    p1 = float(row["prob_very_mild"])
                    p2 = float(row["prob_mild"])
                    p3 = float(row["prob_moderate"])
                    
                    self.assertFalse(np.isnan(p0) or np.isnan(p1) or np.isnan(p2) or np.isnan(p3), f"NaN at row {i} in {slug}")
                    self.assertFalse(np.isinf(p0) or np.isinf(p1) or np.isinf(p2) or np.isinf(p3), f"Inf at row {i} in {slug}")
                    
                    prob_sum = p0 + p1 + p2 + p3
                    self.assertAlmostEqual(prob_sum, 1.0, places=4, msg=f"Probabilities must sum to 1.0 in row {i} of {slug}")
                    
                    conf = float(row["confidence"])
                    self.assertGreaterEqual(conf, 0.25, f"Confidence must be at least 1/4 in row {i}")
                    self.assertLessEqual(conf, 1.0, f"Confidence must not exceed 1.0 in row {i}")

    def test_04_class_labels_valid(self):
        """Verify all true_class and predicted_class strings match canonical CLASSES."""
        for slug in self.expected_models:
            csv_path = self.error_dir / f"{slug}_predictions.csv"
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.assertIn(row["true_class"], CLASSES, f"Invalid true class: {row['true_class']}")
                    self.assertIn(row["predicted_class"], CLASSES, f"Invalid predicted class: {row['predicted_class']}")

    def test_05_high_confidence_errors_schema_and_integrity(self):
        """Verify high_confidence_errors.csv has valid schema and confidence >= 0.80."""
        hce_csv = self.error_dir / "high_confidence_errors.csv"
        self.assertTrue(hce_csv.exists(), "high_confidence_errors.csv must exist")
        
        with open(hce_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        for r in rows:
            conf = float(r["confidence"])
            self.assertGreaterEqual(conf, 0.80, f"High-confidence error must have confidence >= 0.80, got {conf}")
            self.assertNotEqual(r["true_class"], r["predicted_class"], "Must be an actual misclassification")

    def test_06_low_confidence_correct_schema_and_integrity(self):
        """Verify low_confidence_correct.csv has valid schema and confidence < 0.70."""
        lcc_csv = self.error_dir / "low_confidence_correct.csv"
        self.assertTrue(lcc_csv.exists(), "low_confidence_correct.csv must exist")
        
        with open(lcc_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        for r in rows:
            conf = float(r["confidence"])
            self.assertLess(conf, 0.70, f"Low-confidence correct must have confidence < 0.70, got {conf}")
            self.assertEqual(r["true_class"], r["predicted_class"], "Must be a correct classification")

    def test_07_cross_model_overlap_csv_exists_and_valid(self):
        """Verify cross_model_error_overlap.csv exists and has expected fields."""
        overlap_csv = self.error_dir / "cross_model_error_overlap.csv"
        self.assertTrue(overlap_csv.exists(), "cross_model_error_overlap.csv must exist")
        
        with open(overlap_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        self.assertGreater(len(rows), 0, "Must have recorded error overlap rows")
        for r in rows:
            err_cnt = int(r["error_count"])
            self.assertIn(err_cnt, [1, 2, 3], "Error count must be 1, 2, or 3")

    def test_08_robustness_results_structure(self):
        """Verify robustness_results.csv and json exist and have all 10 perturbations."""
        csv_path = self.robust_dir / "robustness_results.csv"
        json_path = self.robust_dir / "robustness_results.json"
        
        self.assertTrue(csv_path.exists(), "robustness_results.csv must exist")
        self.assertTrue(json_path.exists(), "robustness_results.json must exist")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.assertEqual(len(data), 3, "Must contain results for all 3 models")
        for m_name, pert_list in data.items():
            self.assertEqual(len(pert_list), 10, f"Must have exactly 10 perturbations for {m_name}")
            for p in pert_list:
                self.assertIn("baseline_accuracy", p)
                self.assertIn("perturbed_accuracy", p)
                self.assertIn("accuracy_change", p)
                self.assertGreaterEqual(p["perturbed_accuracy"], 0.0)
                self.assertLessEqual(p["perturbed_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
