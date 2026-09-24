"""
Unit tests for programmatic button navigation in the redesigned Streamlit UI.
Verifies that all CTA buttons on the Overview page correctly update session state
and route to the intended destination pages.
"""

import sys
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestButtonNavigation(unittest.TestCase):
    """Test suite verifying all Overview page CTA buttons route correctly."""

    def setUp(self):
        self.app_file = str(PROJECT_ROOT / "app" / "app.py")

    def test_hero_cta_analyze_mri_button(self):
        """Verify hero 'ANALYZE MRI SCAN' button navigates to MRI Analysis page."""
        at = AppTest.from_file(self.app_file, default_timeout=30)
        at.run()
        
        # Verify starting on Overview
        self.assertEqual(at.session_state["nav_radio"], "Overview")
        
        # Click hero analyze button
        at.button(key="hero_btn_analyze").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["nav_radio"], "MRI Analysis")

    def test_hero_cta_explore_results_button(self):
        """Verify hero 'EXPLORE RESULTS' button navigates to Evaluation page."""
        at = AppTest.from_file(self.app_file, default_timeout=30)
        at.run()
        
        # Click hero results button
        at.button(key="hero_btn_results").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["nav_radio"], "Evaluation")

    def test_quick_action_buttons(self):
        """Verify all 4 Quick Action cards navigate to their respective destinations."""
        qa_targets = [
            ("qa_btn_analyze", "MRI Analysis"),
            ("qa_btn_explain", "Explainability"),
            ("qa_btn_compare", "Model Comparison"),
            ("qa_btn_evaluate", "Evaluation"),
        ]

        for btn_key, expected_page in qa_targets:
            at = AppTest.from_file(self.app_file, default_timeout=30)
            at.run()
            self.assertEqual(at.session_state["nav_radio"], "Overview")
            
            at.button(key=btn_key).click().run()
            self.assertEqual(len(at.exception), 0, f"Exception on clicking {btn_key}")
            self.assertEqual(
                at.session_state["nav_radio"],
                expected_page,
                f"Expected {expected_page} on clicking {btn_key}, got {at.session_state['nav_radio']}"
            )

    def test_model_card_inspect_buttons(self):
        """Verify all 3 candidate model cards navigate to Model Comparison page."""
        model_btn_keys = ["btn_inspect_mb", "btn_inspect_eff", "btn_inspect_res"]
        
        for btn_key in model_btn_keys:
            at = AppTest.from_file(self.app_file, default_timeout=30)
            at.run()
            self.assertEqual(at.session_state["nav_radio"], "Overview")
            
            at.button(key=btn_key).click().run()
            self.assertEqual(len(at.exception), 0, f"Exception on clicking {btn_key}")
            self.assertEqual(at.session_state["nav_radio"], "Model Comparison")


if __name__ == "__main__":
    unittest.main()
