"""
Data Loader Utilities for Metrics, Visualizations, and Test Set Manifests.
Reads precomputed evaluation results and test catalogs safely without external binary dependencies.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data
def load_final_model_comparison() -> List[Dict[str, Any]]:
    """Load held-out test set multi-model comparison table from CSV."""
    csv_path = PROJECT_ROOT / "results" / "metrics" / "final_model_comparison.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


@st.cache_data
def load_final_test_results() -> Dict[str, Any]:
    """Load detailed test evaluation metrics from JSON."""
    json_path = PROJECT_ROOT / "results" / "metrics" / "final_test_results.json"
    if not json_path.exists():
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_gradcam_catalog() -> Dict[str, Any]:
    """Load pre-generated Grad-CAM explainability catalog."""
    json_path = PROJECT_ROOT / "results" / "metrics" / "gradcam_results.json"
    if not json_path.exists():
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_calibration_results() -> Dict[str, Any]:
    """Load calibration and reliability analysis results from JSON."""
    json_path = PROJECT_ROOT / "results" / "metrics" / "calibration_results.json"
    if not json_path.exists():
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_error_analysis_summary() -> Dict[str, Any]:
    """Load aggregated error analysis summary from JSON."""
    json_path = PROJECT_ROOT / "results" / "error_analysis" / "error_analysis_summary.json"
    if not json_path.exists():
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_robustness_results() -> Dict[str, Any]:
    """Load controlled perturbation robustness results from JSON."""
    json_path = PROJECT_ROOT / "results" / "robustness" / "robustness_results.json"
    if not json_path.exists():
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_high_confidence_errors() -> List[Dict[str, Any]]:
    """Load high-confidence error cases from CSV."""
    csv_path = PROJECT_ROOT / "results" / "error_analysis" / "high_confidence_errors.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


@st.cache_data
def load_cross_model_error_overlap() -> List[Dict[str, Any]]:
    """Load cross-model error overlap cases from CSV."""
    csv_path = PROJECT_ROOT / "results" / "error_analysis" / "cross_model_error_overlap.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


@st.cache_data
def get_sample_test_images() -> Dict[str, List[Dict[str, str]]]:
    """
    Provide curated sample MRI scans from each class from the held-out test split
    for immediate one-click testing in the MRI Analysis demo page.
    Resolves from lightweight deployment test_samples directory first, with fallback to local dataset.
    """
    def _resolve(cls_subdir: str, filename: str) -> str:
        demo_p = PROJECT_ROOT / "data" / "test_samples" / filename
        if demo_p.exists():
            return str(demo_p)
        dataset_p = PROJECT_ROOT / "data" / "dataset" / cls_subdir / filename
        return str(dataset_p)

    samples_by_class = {
        "Non-Demented": [
            {"filename": "non_2751.jpg", "path": _resolve("Non_Demented", "non_2751.jpg"), "label": "Non-Demented"},
            {"filename": "non_1659.jpg", "path": _resolve("Non_Demented", "non_1659.jpg"), "label": "Non-Demented"},
            {"filename": "non_1263.jpg", "path": _resolve("Non_Demented", "non_1263.jpg"), "label": "Non-Demented (Borderline)"}
        ],
        "Very Mild Demented": [
            {"filename": "verymild_1576.jpg", "path": _resolve("Very_Mild_Demented", "verymild_1576.jpg"), "label": "Very Mild Demented"},
            {"filename": "verymild_2177.jpg", "path": _resolve("Very_Mild_Demented", "verymild_2177.jpg"), "label": "Very Mild Demented"},
            {"filename": "verymild_1664.jpg", "path": _resolve("Very_Mild_Demented", "verymild_1664.jpg"), "label": "Very Mild Demented (Subtle)"}
        ],
        "Mild Demented": [
            {"filename": "mild_71.jpg", "path": _resolve("Mild_Demented", "mild_71.jpg"), "label": "Mild Demented"},
            {"filename": "mild_821.jpg", "path": _resolve("Mild_Demented", "mild_821.jpg"), "label": "Mild Demented"},
            {"filename": "mild_33.jpg", "path": _resolve("Mild_Demented", "mild_33.jpg"), "label": "Mild Demented (Early transition)"}
        ],
        "Moderate Demented": [
            {"filename": "moderate_27.jpg", "path": _resolve("Moderate_Demented", "moderate_27.jpg"), "label": "Moderate Demented"},
            {"filename": "moderate_53.jpg", "path": _resolve("Moderate_Demented", "moderate_53.jpg"), "label": "Moderate Demented"},
            {"filename": "moderate_9.jpg", "path": _resolve("Moderate_Demented", "moderate_9.jpg"), "label": "Moderate Demented"}
        ]
    }
    return samples_by_class

