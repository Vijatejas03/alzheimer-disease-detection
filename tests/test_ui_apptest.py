"""
Automated Test Suite for Streamlit UI and Navigation Routing.
Uses Streamlit AppTest to verify all 8 pages render cleanly without exceptions.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestStreamlitUIPages(unittest.TestCase):
    """Test suite verifying Streamlit pages execute without exceptions."""

    def test_all_8_pages_render_without_exception(self):
        """Verify all 8 pages cycle without unhandled exceptions or error alerts."""
        from streamlit.testing.v1 import AppTest
        
        app_file = str(PROJECT_ROOT / "app" / "app.py")
        at = AppTest.from_file(app_file, default_timeout=35)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Exceptions on initial load: {at.exception}")

        pages = [
            "Overview",
            "MRI Analysis",
            "Model Comparison",
            "Evaluation",
            "Explainability",
            "Error & Robustness",
            "Methodology",
            "About"
        ]

        for page_name in pages:
            at.sidebar.radio[0].set_value(page_name).run()
            self.assertEqual(
                len(at.exception), 0,
                f"Exception encountered on page '{page_name}': {[e.value for e in at.exception]}"
            )
            self.assertEqual(
                len(at.error), 0,
                f"Error alert encountered on page '{page_name}': {[err.value for err in at.error]}"
            )

    def test_navigation_alias_routing(self):
        """Verify legacy aliases route safely without crashing."""
        from app.components.home_page import render_home_page
        from app.components.mri_analysis_page import render_mri_analysis_page
        from app.components.model_comparison_page import render_model_comparison_page
        from app.components.evaluation_results_page import render_evaluation_results_page
        from app.components.gradcam_gallery_page import render_gradcam_gallery_page
        from app.components.error_robustness_page import render_error_robustness_page
        from app.components.methodology_page import render_methodology_page
        from app.components.about_page import render_about_page
        
        # Verify all component callables exist and are functions
        self.assertTrue(callable(render_home_page))
        self.assertTrue(callable(render_mri_analysis_page))
        self.assertTrue(callable(render_model_comparison_page))
        self.assertTrue(callable(render_evaluation_results_page))
        self.assertTrue(callable(render_gradcam_gallery_page))
        self.assertTrue(callable(render_error_robustness_page))
        self.assertTrue(callable(render_methodology_page))
        self.assertTrue(callable(render_about_page))

    def test_three_model_agreement_and_gradcam_flow(self):
        """Verify 3-model agreement engine and Grad-CAM execution flow."""
        from PIL import Image
        from app.utils.inference_engine import (
            load_cached_model,
            run_model_inference,
            preprocess_image_for_inference,
            generate_gradcam
        )
        from app.utils.data_loader import get_sample_test_images
        
        samples = get_sample_test_images()
        img_path = samples["Mild Demented"][0]["path"]
        pil_img = Image.open(img_path)
        tensor, display_img, meta = preprocess_image_for_inference(pil_img)
        
        models = {}
        results = {}
        for m_name in ["MobileNetV2", "EfficientNet-B0", "ResNet18"]:
            m_obj, msg = load_cached_model(m_name)
            self.assertIsNotNone(m_obj, f"Model {m_name} failed to load: {msg}")
            models[m_name] = m_obj
            results[m_name] = run_model_inference(m_obj, tensor)
            self.assertIn("predicted_class", results[m_name])
            self.assertIn("confidence", results[m_name])
            self.assertGreater(results[m_name]["confidence"], 0.5)

        # Verify agreement calculation
        preds = [results[m]["predicted_class"] for m in results]
        self.assertEqual(len(preds), 3)
        self.assertEqual(len(set(preds)), 1, "Expected unanimous agreement on high-confidence Mild Demented sample")

        # Verify Grad-CAM calculation
        eff_model = models["EfficientNet-B0"]
        eff_pred_idx = results["EfficientNet-B0"]["predicted_idx"]
        cam_np, ov, target_layer_desc = generate_gradcam(eff_model, "EfficientNet-B0", tensor, display_img, eff_pred_idx)
        self.assertEqual(cam_np.shape, (224, 224))
        self.assertEqual(ov.size, (224, 224))
        self.assertIn("features.8", target_layer_desc)

    def test_mri_analysis_model_switching_and_parameter_display(self):
        """Verify MRI Analysis page interactive flow and parameter counts for all 3 models."""
        from streamlit.testing.v1 import AppTest

        app_file = str(PROJECT_ROOT / "app" / "app.py")
        at = AppTest.from_file(app_file, default_timeout=40)
        at.run()
        
        # Navigate to MRI Analysis
        at.sidebar.radio[0].set_value("MRI Analysis").run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

        # Select Curated Research Sample
        at.radio[0].set_value("Load Curated Research Sample").run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

        expected_param_counts = {
            "MobileNetV2": "2,228,996",
            "EfficientNet-B0": "4,012,672",
            "ResNet18": "11,178,564"
        }

        for model_name, expected_param_str in expected_param_counts.items():
            # Select model in primary architecture selectbox (index 2: [stage, sample, model])
            at.selectbox[2].set_value(model_name).run()
            self.assertEqual(
                len(at.exception), 0,
                f"Exception encountered for model '{model_name}': {[e.value for e in at.exception]}"
            )
            self.assertEqual(
                len(at.error), 0,
                f"Error alert encountered for model '{model_name}': {[err.value for err in at.error]}"
            )
            all_md = " ".join([m.value for m in at.markdown])
            self.assertIn(
                expected_param_str, all_md,
                f"Expected parameter count '{expected_param_str}' not found in rendered markdown for {model_name}"
            )
            # Verify Grad-CAM user guide elements
            self.assertIn("How to Read This Explanation", all_md)
            self.assertIn("Attribution Color Legend", all_md)
            self.assertIn("model attribution", all_md)
            self.assertIn("not disease severity or confirmed brain pathology", all_md)
            self.assertIn("Grad-CAM is an explainability visualization.", all_md)

    def test_out_of_domain_upload_intercepted_and_suppresses_inference(self):
        """Verify that uploading an out-of-domain image displays Input rejected and halts inference completely."""
        import io
        from PIL import Image
        from streamlit.testing.v1 import AppTest

        # Create simulated non-brain photo bytes (e.g. dog/cat/human color photo)
        ood_img = Image.new("RGB", (160, 160), (180, 120, 60))
        bio = io.BytesIO()
        ood_img.save(bio, format="PNG")
        ood_bytes = bio.getvalue()

        app_file = str(PROJECT_ROOT / "app" / "app.py")
        at = AppTest.from_file(app_file, default_timeout=40)
        at.run()

        # Navigate to MRI Analysis
        at.sidebar.radio[0].set_value("MRI Analysis").run()
        self.assertEqual(len(at.exception), 0)

        # Select 'Upload Brain MRI Scan'
        at.radio[0].set_value("Upload Brain MRI Scan").run()
        self.assertEqual(len(at.exception), 0)

        # Upload non-brain out-of-domain file
        at.file_uploader[0].upload("unrelated_photo.png", ood_bytes).run()

        # Assert no unhandled exceptions
        self.assertEqual(len(at.exception), 0)

        # Assert warning alert displays exact required text
        all_text = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning])
        self.assertIn("Input rejected", all_text)
        self.assertIn("This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.", all_text)
        self.assertIn("exclusively on axial brain MRI scans", all_text)
        self.assertIn("Input Domain Policy & Requirements", all_text)

        # Assert inference was completely suppressed
        self.assertNotIn("04 Deep Learning Inference", all_text)
        self.assertNotIn("05 Multi-Class Probability Distribution", all_text)
        self.assertNotIn("06 Mathematical Uncertainty Analysis", all_text)
        self.assertNotIn("07 3-Model Consensus Agreement Engine", all_text)
        self.assertNotIn("08 Grad-CAM Explainability", all_text)


if __name__ == "__main__":
    unittest.main()


