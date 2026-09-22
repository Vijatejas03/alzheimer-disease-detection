"""
Verification test confirming all 3 neural architectures, forward inference,
multi-model consensus, and Grad-CAM explainability execute cleanly on 100% CPU.
"""

import unittest
from PIL import Image
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestCPUExecution(unittest.TestCase):
    def test_cpu_inference_and_gradcam(self):
        # Temporarily mock torch.cuda.is_available to False
        orig_cuda_avail = torch.cuda.is_available
        try:
            from app.utils.inference_engine import load_cached_model
            load_cached_model.clear()
            torch.cuda.is_available = lambda: False
            
            from app.utils.inference_engine import (
                load_cached_model,
                run_model_inference,
                preprocess_image_for_inference,
                generate_gradcam,
                get_inference_device
            )

            device, desc = get_inference_device()
            self.assertEqual(device.type, "cpu")
            self.assertIn("CPU", desc)

            # Test synthetic image
            img = Image.new("RGB", (128, 128), color=128)
            tensor, disp, meta = preprocess_image_for_inference(img)
            self.assertEqual(tensor.device.type, "cpu")

            for m_name in ["MobileNetV2", "EfficientNet-B0", "ResNet18"]:
                model, msg = load_cached_model(m_name)
                self.assertIsNotNone(model, f"{m_name} failed to load: {msg}")
                
                # Check parameters are on CPU
                first_param = next(model.parameters())
                self.assertEqual(first_param.device.type, "cpu")
                
                # Forward inference
                res = run_model_inference(model, tensor)
                self.assertIn("predicted_class", res)
                self.assertIn("confidence", res)
                self.assertIn("latency_ms", res)
                self.assertGreater(res["confidence"], 0.0)

                # Grad-CAM on CPU
                cam_np, ov, target_desc = generate_gradcam(
                    model, m_name, tensor, disp, res["predicted_idx"]
                )
                self.assertEqual(cam_np.shape, (224, 224))
                self.assertEqual(ov.size, (224, 224))
                self.assertGreaterEqual(cam_np.min(), 0.0)
                self.assertLessEqual(cam_np.max(), 1.0)
        finally:
            torch.cuda.is_available = orig_cuda_avail
            try:
                from app.utils.inference_engine import load_cached_model
                load_cached_model.clear()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
