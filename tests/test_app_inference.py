"""
Automated Test Suite for Streamlit Application Inference and Explainability Engine.
Verifies checkpoint loading, output dimensionality, probability summation,
forward pass inference, and Grad-CAM generation across all candidate architectures.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.inference_engine import (
    MODEL_CONFIGS,
    CLASS_NAMES,
    load_cached_model,
    preprocess_image_for_inference,
    run_model_inference,
    generate_gradcam,
    get_inference_device
)
from app.utils.data_loader import (
    load_final_model_comparison,
    load_final_test_results,
    load_gradcam_catalog,
    get_sample_test_images
)


class TestStreamlitAppInference(unittest.TestCase):
    """Test suite verifying Streamlit application inference modules."""

    @classmethod
    def setUpClass(cls):
        cls.device, cls.device_desc = get_inference_device()
        print(f"\n[Test Suite] Initialized on device: {cls.device_desc}")
        
        # Load a real sample image from the test set
        sample_dict = get_sample_test_images()
        cls.test_img_path = sample_dict["Mild Demented"][0]["path"]
        assert Path(cls.test_img_path).exists(), f"Sample image missing: {cls.test_img_path}"
        cls.raw_pil_img = Image.open(cls.test_img_path)

    def test_01_sample_data_loader(self):
        """Verify sample test image loader returns valid files for all 4 classes."""
        samples = get_sample_test_images()
        self.assertEqual(len(samples), 4, "Must contain exactly 4 stage categories")
        for stage, scan_list in samples.items():
            self.assertGreaterEqual(len(scan_list), 1, f"Must have at least 1 sample for {stage}")
            for scan in scan_list:
                self.assertTrue(Path(scan["path"]).exists(), f"File does not exist: {scan['path']}")

    def test_02_metrics_and_catalogs_exist(self):
        """Verify that precomputed benchmark results load properly."""
        comparison_rows = load_final_model_comparison()
        self.assertEqual(len(comparison_rows), 3, "Comparison table must contain 3 models")
        
        test_results = load_final_test_results()
        self.assertIn("models_evaluated", test_results)
        self.assertEqual(len(test_results["models_evaluated"]), 3)
        
        gradcam_cat = load_gradcam_catalog()
        self.assertIn("models", gradcam_cat)
        self.assertIn("summary", gradcam_cat)
        self.assertGreaterEqual(gradcam_cat["summary"]["total_examples_generated"], 30)

    def test_03_preprocessing_pipeline(self):
        """Verify image preprocessing produces standardized 224x224 RGB tensor and meta."""
        tensor, display_img, meta = preprocess_image_for_inference(self.raw_pil_img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224), "Preprocessed tensor shape must be [1, 3, 224, 224]")
        self.assertEqual(display_img.size, (224, 224), "Display image must be resized to 224x224")
        self.assertEqual(display_img.mode, "RGB", "Display image must be RGB")
        self.assertIn("original_width", meta)
        self.assertIn("original_height", meta)

    def test_04_mobilenet_v2_inference_and_gradcam(self):
        """Verify MobileNetV2 checkpoint loading, inference, 4 classes, and Grad-CAM."""
        model, msg = load_cached_model("MobileNetV2")
        self.assertIsNotNone(model, f"MobileNetV2 failed to load: {msg}")
        
        tensor, display_img, _ = preprocess_image_for_inference(self.raw_pil_img)
        res = run_model_inference(model, tensor)
        
        # Verify 4 classes
        self.assertEqual(len(res["probabilities"]), 4)
        for c in CLASS_NAMES:
            self.assertIn(c, res["probabilities"])
            
        # Verify probabilities sum to 1.0
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4, msg="Probabilities must sum to 1.0")
        
        # Verify confidence matches predicted class probability
        pred_class = res["predicted_class"]
        self.assertAlmostEqual(res["confidence"], res["probabilities"][pred_class], places=4)
        
        # Verify Grad-CAM generation
        cam_np, overlay_img, target_layer = generate_gradcam(
            model=model,
            model_name="MobileNetV2",
            input_tensor=tensor,
            display_img=display_img,
            target_class=res["predicted_idx"]
        )
        self.assertEqual(cam_np.shape, (224, 224))
        self.assertGreater(cam_np.max(), 0.0, "CAM heatmap must not be all zeros")
        self.assertEqual(overlay_img.size, (224, 224))
        self.assertIn("features.18", target_layer)

    def test_05_efficientnet_b0_inference_and_gradcam(self):
        """Verify EfficientNet-B0 checkpoint loading, inference, 4 classes, and Grad-CAM."""
        model, msg = load_cached_model("EfficientNet-B0")
        self.assertIsNotNone(model, f"EfficientNet-B0 failed to load: {msg}")
        
        tensor, display_img, _ = preprocess_image_for_inference(self.raw_pil_img)
        res = run_model_inference(model, tensor)
        
        # Verify 4 classes
        self.assertEqual(len(res["probabilities"]), 4)
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)
        
        # Verify Grad-CAM generation
        cam_np, overlay_img, target_layer = generate_gradcam(
            model=model,
            model_name="EfficientNet-B0",
            input_tensor=tensor,
            display_img=display_img,
            target_class=res["predicted_idx"]
        )
        self.assertEqual(cam_np.shape, (224, 224))
        self.assertGreater(cam_np.max(), 0.0)
        self.assertEqual(overlay_img.size, (224, 224))
        self.assertIn("features.8", target_layer)

    def test_06_resnet18_inference_and_gradcam(self):
        """Verify ResNet-18 checkpoint loading, inference, 4 classes, and Grad-CAM."""
        model, msg = load_cached_model("ResNet18")
        self.assertIsNotNone(model, f"ResNet18 failed to load: {msg}")
        
        tensor, display_img, _ = preprocess_image_for_inference(self.raw_pil_img)
        res = run_model_inference(model, tensor)
        
        # Verify 4 classes
        self.assertEqual(len(res["probabilities"]), 4)
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)
        
        # Verify Grad-CAM generation
        cam_np, overlay_img, target_layer = generate_gradcam(
            model=model,
            model_name="ResNet18",
            input_tensor=tensor,
            display_img=display_img,
            target_class=res["predicted_idx"]
        )
        self.assertEqual(cam_np.shape, (224, 224))
        self.assertGreater(cam_np.max(), 0.0)
        self.assertEqual(overlay_img.size, (224, 224))
        self.assertIn("layer4.1", target_layer)


if __name__ == "__main__":
    unittest.main()
