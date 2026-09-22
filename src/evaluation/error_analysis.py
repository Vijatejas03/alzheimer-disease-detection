"""
Comprehensive Error Analysis Engine for Multi-Stage Alzheimer's MRI Classification.
Performs prediction-level logging, class-wise performance breakdown, confusion matrix analysis,
confidence vs. uncertainty diagnostics, high-confidence error extraction, low-confidence correct
identification, cross-model error overlap analysis, and Grad-CAM visual error inspection.

Strict Research Integrity Guarantee:
- Evaluates strictly on the untouched 960-image held-out test split.
- Preserves all model weights, splits, and official benchmark metrics.
- Uses pure PyTorch, NumPy, and Matplotlib without external C-extension risks.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.model_factory import load_trained_model, MODEL_REGISTRY
from src.data.dataset import AlzheimerMRISplitDataset
from src.data.augmentation import get_validation_transforms
from src.data.validation import CLASSES, CLASS_TO_IDX, CLASS_DISPLAY_NAMES
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.architectures import get_target_convolutional_layer
from src.utils.logger import logger


MODEL_KEYS = [
    ("MobileNetV2", "mobilenet_v2", "results/models/mobilenet_v2_best.pt", "results/calibration/mobilenet_v2_temperature.json"),
    ("EfficientNet-B0", "efficientnet_b0", "results/models/efficientnet_b0_best.pt", "results/calibration/efficientnet_b0_temperature.json"),
    ("ResNet18", "resnet18", "results/models/resnet18_best.pt", "results/calibration/resnet18_temperature.json")
]


def load_temperature(temp_json_path: Path) -> float:
    """Load learned scalar temperature if available, else default to 1.0."""
    if temp_json_path.exists():
        try:
            with open(temp_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return float(data.get("learned_temperature", 1.0))
        except Exception as e:
            logger.warning(f"Error loading temperature from {temp_json_path}: {e}")
    return 1.0


def compute_normalized_entropy(probs: np.ndarray, num_classes: int = 4, eps: float = 1e-12) -> float:
    """Compute normalized Shannon entropy in [0.0, 1.0]."""
    probs = np.clip(probs, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(num_classes)
    return float(entropy / max_entropy)


# =====================================================================
# 1. PREDICTION EXTRACTION PER MODEL
# =====================================================================

def extract_model_predictions(
    model: nn.Module,
    test_dataset: AlzheimerMRISplitDataset,
    temperature: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> List[Dict[str, Any]]:
    """
    Run forward pass on all test samples and record detailed prediction records.
    """
    model.to(device)
    model.eval()
    
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    records = []
    
    sample_idx = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
            # Calibrated probabilities
            cal_logits = logits / max(1e-4, temperature)
            cal_probs = F.softmax(cal_logits, dim=1).cpu().numpy()
            
            targets_np = targets.numpy()
            batch_size = images.size(0)
            
            for b in range(batch_size):
                p_vec = probs[b]
                cal_p_vec = cal_probs[b]
                t_idx = int(targets_np[b])
                
                # Sorted probability indices descending
                sorted_indices = np.argsort(-p_vec)
                pred_idx = int(sorted_indices[0])
                second_idx = int(sorted_indices[1])
                
                confidence = float(p_vec[pred_idx])
                second_highest = float(p_vec[second_idx])
                margin = float(confidence - second_highest)
                entropy_val = compute_normalized_entropy(p_vec, num_classes=4)
                cal_confidence = float(cal_p_vec[pred_idx])
                
                img_path, _ = test_dataset.samples[sample_idx]
                rel_path = os.path.relpath(img_path, PROJECT_ROOT).replace("\\", "/")
                
                records.append({
                    "sample_idx": sample_idx,
                    "image_path": rel_path,
                    "filename": os.path.basename(img_path),
                    "true_class": CLASSES[t_idx],
                    "true_idx": t_idx,
                    "predicted_class": CLASSES[pred_idx],
                    "predicted_idx": pred_idx,
                    "is_correct": bool(pred_idx == t_idx),
                    "confidence": confidence,
                    "second_highest_prob": second_highest,
                    "prediction_margin": margin,
                    "entropy_uncertainty": entropy_val,
                    "calibrated_confidence": cal_confidence,
                    "prob_non_demented": float(p_vec[0]),
                    "prob_very_mild": float(p_vec[1]),
                    "prob_mild": float(p_vec[2]),
                    "prob_moderate": float(p_vec[3])
                })
                sample_idx += 1
                
    return records


# =====================================================================
# 2. CLASS-WISE METRICS & CONFUSION ANALYSIS
# =====================================================================

def compute_class_wise_error_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute total samples, correct, incorrect, accuracy, precision, recall,
    specificity, F1, false positives, and false negatives for all 4 classes.
    """
    cm = np.zeros((4, 4), dtype=int)
    for r in records:
        cm[r["true_idx"], r["predicted_idx"]] += 1
        
    total_samples = len(records)
    class_metrics = {}
    
    for c_idx, c_name in enumerate(CLASSES):
        tp = int(cm[c_idx, c_idx])
        fn = int(np.sum(cm[c_idx, :]) - tp)
        fp = int(np.sum(cm[:, c_idx]) - tp)
        tn = int(total_samples - tp - fn - fp)
        support = tp + fn
        
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = float(tp / support) if support > 0 else 0.0
        
        class_metrics[c_name] = {
            "class_idx": c_idx,
            "total_samples": support,
            "correct_predictions": tp,
            "incorrect_predictions": fn,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1_score": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }
        
    # Extract non-zero confusion pairs
    confusion_pairs = []
    for t in range(4):
        t_support = int(np.sum(cm[t, :]))
        for p in range(4):
            if t != p and cm[t, p] > 0:
                count = int(cm[t, p])
                pct = float((count / max(1, t_support)) * 100.0)
                confusion_pairs.append({
                    "true_class": CLASSES[t],
                    "predicted_class": CLASSES[p],
                    "count": count,
                    "true_class_support": t_support,
                    "percentage_of_true_class": pct
                })
                
    # Sort confusion pairs by frequency descending
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_pairs": confusion_pairs
    }


# =====================================================================
# 3. CONFIDENCE & UNCERTAINTY STATS (CORRECT VS. INCORRECT)
# =====================================================================

def compute_confidence_uncertainty_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate mean, median, min, max confidence and uncertainty for correct vs. incorrect."""
    correct_confs = [r["confidence"] for r in records if r["is_correct"]]
    incorrect_confs = [r["confidence"] for r in records if not r["is_correct"]]
    
    correct_ents = [r["entropy_uncertainty"] for r in records if r["is_correct"]]
    incorrect_ents = [r["entropy_uncertainty"] for r in records if not r["is_correct"]]
    
    def get_stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        arr = np.array(vals)
        return {
            "count": len(vals),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr))
        }
        
    return {
        "correct_confidence": get_stats(correct_confs),
        "incorrect_confidence": get_stats(incorrect_confs),
        "correct_uncertainty": get_stats(correct_ents),
        "incorrect_uncertainty": get_stats(incorrect_ents)
    }


# =====================================================================
# 4. PLOTTING CONFIDENCE DISTRIBUTIONS
# =====================================================================

def plot_error_confidence_distribution(
    records: List[Dict[str, Any]],
    model_name: str,
    save_path: str
) -> None:
    """
    Generate high-contrast dual-panel distribution plot of confidence and uncertainty
    comparing correct predictions vs. incorrect predictions.
    """
    correct_confs = [r["confidence"] for r in records if r["is_correct"]]
    incorrect_confs = [r["confidence"] for r in records if not r["is_correct"]]
    
    correct_ents = [r["entropy_uncertainty"] for r in records if r["is_correct"]]
    incorrect_ents = [r["entropy_uncertainty"] for r in records if not r["is_correct"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    
    # 1. Confidence Distribution
    ax1.set_facecolor('#F8FAFC')
    bins_conf = np.linspace(0.25, 1.0, 31)
    
    ax1.hist(
        correct_confs, bins=bins_conf, color='#0F4C81', alpha=0.75,
        edgecolor='#0A2540', label=f'Correct ({len(correct_confs)})'
    )
    if incorrect_confs:
        ax1.hist(
            incorrect_confs, bins=bins_conf, color='#DC2626', alpha=0.85,
            edgecolor='#7F1D1D', label=f'Incorrect ({len(incorrect_confs)})'
        )
        
    mean_corr = np.mean(correct_confs) if correct_confs else 0.0
    mean_inc = np.mean(incorrect_confs) if incorrect_confs else 0.0
    med_inc = np.median(incorrect_confs) if incorrect_confs else 0.0
    
    stats_text = (
        f"Correct Mean Conf: {mean_corr*100:.1f}%\n"
        f"Incorrect Mean Conf: {mean_inc*100:.1f}%\n"
        f"Incorrect Median Conf: {med_inc*100:.1f}%\n"
        f"Total Errors: {len(incorrect_confs)} / {len(records)}"
    )
    ax1.text(
        0.05, 0.95, stats_text, transform=ax1.transAxes,
        fontsize=9.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFFF', edgecolor='#CBD5E1', alpha=0.95)
    )
    
    ax1.set_title(f'{model_name} — Confidence Distribution (Correct vs. Incorrect)', fontsize=11.5, fontweight='bold', color='#0F4C81')
    ax1.set_xlabel('Predicted Confidence (max softmax probability)', fontsize=10.5, fontweight='bold', color='#1E293B')
    ax1.set_ylabel('Sample Count', fontsize=10.5, fontweight='bold', color='#1E293B')
    ax1.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax1.legend(loc='upper right', fontsize=9.5)
    
    # 2. Entropy / Uncertainty Distribution
    ax2.set_facecolor('#F8FAFC')
    bins_ent = np.linspace(0.0, 1.0, 31)
    
    ax2.hist(
        correct_ents, bins=bins_ent, color='#0D9488', alpha=0.75,
        edgecolor='#042F2E', label=f'Correct ({len(correct_ents)})'
    )
    if incorrect_ents:
        ax2.hist(
            incorrect_ents, bins=bins_ent, color='#DC2626', alpha=0.85,
            edgecolor='#7F1D1D', label=f'Incorrect ({len(incorrect_ents)})'
        )
        
    mean_ent_corr = np.mean(correct_ents) if correct_ents else 0.0
    mean_ent_inc = np.mean(incorrect_ents) if incorrect_ents else 0.0
    
    ent_text = (
        f"Correct Mean Entropy: {mean_ent_corr:.3f}\n"
        f"Incorrect Mean Entropy: {mean_ent_inc:.3f}\n"
        f"Separation: {abs(mean_ent_inc - mean_ent_corr):.3f}"
    )
    ax2.text(
        0.55, 0.95, ent_text, transform=ax2.transAxes,
        fontsize=9.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFFF', edgecolor='#CBD5E1', alpha=0.95)
    )
    
    ax2.set_title(f'{model_name} — Normalized Shannon Entropy Uncertainty', fontsize=11.5, fontweight='bold', color='#0D9488')
    ax2.set_xlabel('Normalized Entropy (0 = Certain, 1 = Maximal Ambiguity)', fontsize=10.5, fontweight='bold', color='#1E293B')
    ax2.set_ylabel('Sample Count', fontsize=10.5, fontweight='bold', color='#1E293B')
    ax2.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax2.legend(loc='upper right', fontsize=9.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved error confidence plot to {save_path}")


# =====================================================================
# 5. CROSS-MODEL ERROR OVERLAP
# =====================================================================

def analyze_cross_model_overlap(
    model_predictions: Dict[str, List[Dict[str, Any]]],
    save_csv_path: str,
    save_fig_path: str
) -> Dict[str, Any]:
    """
    Compare test predictions across MobileNetV2, EfficientNet-B0, and ResNet-18.
    Calculates errors unique to 1 model, shared by 2 models, and shared by all 3 models.
    """
    models = list(model_predictions.keys())
    num_samples = len(next(iter(model_predictions.values())))
    
    # Track error sets by relative image path
    error_sets = {}
    for m in models:
        errs = set()
        for r in model_predictions[m]:
            if not r["is_correct"]:
                errs.add(r["image_path"])
        error_sets[m] = errs
        
    m1, m2, m3 = models[0], models[1], models[2]
    e1, e2, e3 = error_sets[m1], error_sets[m2], error_sets[m3]
    
    all_three = e1.intersection(e2).intersection(e3)
    exactly_two_12 = (e1.intersection(e2)) - e3
    exactly_two_23 = (e2.intersection(e3)) - e1
    exactly_two_13 = (e1.intersection(e3)) - e2
    exactly_two_total = exactly_two_12.union(exactly_two_23).union(exactly_two_13)
    
    only_m1 = e1 - e2 - e3
    only_m2 = e2 - e1 - e3
    only_m3 = e3 - e1 - e2
    exactly_one_total = only_m1.union(only_m2).union(only_m3)
    
    total_union_errors = e1.union(e2).union(e3)
    correct_all_three_count = num_samples - len(total_union_errors)
    
    overlap_rows = []
    # Build per-sample overlap table
    for i in range(num_samples):
        path = model_predictions[m1][i]["image_path"]
        true_cls = model_predictions[m1][i]["true_class"]
        
        m1_pred = model_predictions[m1][i]["predicted_class"]
        m2_pred = model_predictions[m2][i]["predicted_class"]
        m3_pred = model_predictions[m3][i]["predicted_class"]
        
        m1_err = not model_predictions[m1][i]["is_correct"]
        m2_err = not model_predictions[m2][i]["is_correct"]
        m3_err = not model_predictions[m3][i]["is_correct"]
        
        err_count = int(m1_err) + int(m2_err) + int(m3_err)
        
        if err_count > 0:
            overlap_rows.append({
                "image_path": path,
                "filename": os.path.basename(path),
                "true_class": true_cls,
                f"{m1}_pred": m1_pred,
                f"{m1}_error": m1_err,
                f"{m2}_pred": m2_pred,
                f"{m2}_error": m2_err,
                f"{m3}_pred": m3_pred,
                f"{m3}_error": m3_err,
                "error_count": err_count,
                "category": "All 3 Failed" if err_count == 3 else ("Shared by 2" if err_count == 2 else "Unique to 1")
            })
            
    # Sort by error_count descending
    overlap_rows.sort(key=lambda x: x["error_count"], reverse=True)
    
    # Save CSV
    if overlap_rows:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        with open(save_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(overlap_rows[0].keys()))
            writer.writeheader()
            writer.writerows(overlap_rows)
            
    # Generate visualization
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    categories = [
        "Correct in All 3",
        "Unique to 1 Model",
        "Shared by Exactly 2",
        "Misclassified by All 3"
    ]
    counts = [
        correct_all_three_count,
        len(exactly_one_total),
        len(exactly_two_total),
        len(all_three)
    ]
    colors = ['#10B981', '#3B82F6', '#F59E0B', '#DC2626']
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='#0F172A', width=0.55, alpha=0.9)
    for b in bars:
        h = b.get_height()
        pct = (h / num_samples) * 100.0
        ax.annotate(
            f'{h} ({pct:.1f}%)',
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1E293B'
        )
        
    ax.set_title("Cross-Architecture Error Overlap on Held-Out Test Set (N = 960)", fontsize=12.5, fontweight='bold', color='#0F4C81')
    ax.set_ylabel("Number of Test Images", fontsize=11, fontweight='bold', color='#1E293B')
    ax.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax.set_ylim([0, max(counts) * 1.15])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cross-model error overlap plot to {save_fig_path}")
    
    return {
        "total_test_samples": num_samples,
        "correct_all_three": correct_all_three_count,
        "misclassified_all_three": len(all_three),
        "misclassified_exactly_two": len(exactly_two_total),
        "misclassified_exactly_one": len(exactly_one_total),
        "unique_errors": {
            m1: len(only_m1),
            m2: len(only_m2),
            m3: len(only_m3)
        },
        "all_three_error_paths": list(all_three)
    }


# =====================================================================
# 6. GRAD-CAM FOR HIGH-CONFIDENCE ERRORS
# =====================================================================

def generate_gradcam_for_representative_errors(
    model: nn.Module,
    slug: str,
    high_conf_errors: List[Dict[str, Any]],
    output_dir: str,
    max_cases: int = 4
) -> List[str]:
    """
    Generate Grad-CAM heatmaps for representative high-confidence errors.
    Shows Original MRI, Heatmap, and Overlay side-by-side.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    target_layer = get_target_convolutional_layer(model, slug)
    cam_engine = GradCAM(model, target_layer=target_layer, model_name=slug)
    transform = get_validation_transforms(img_size=(224, 224))
    
    saved_figures = []
    # Pick up to max_cases deterministically
    cases_to_plot = high_conf_errors[:max_cases]
    
    for idx, case in enumerate(cases_to_plot):
        img_abs = os.path.join(PROJECT_ROOT, case["image_path"])
        if not os.path.exists(img_abs):
            continue
            
        raw_pil = Image.open(img_abs).convert('RGB')
        display_img = raw_pil.resize((224, 224), Image.Resampling.BILINEAR)
        input_tensor = transform(raw_pil).unsqueeze(0).to(device)
        
        pred_idx = CLASS_TO_IDX[case["predicted_class"]]
        cam_np, _, _ = cam_engine.generate_cam(input_tensor, target_class=pred_idx)
        overlay_img = overlay_heatmap(display_img, cam_np, alpha=0.45, colormap_name='jet')
        
        # Plot 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=250)
        fig.patch.set_facecolor('#FFFFFF')
        
        axes[0].imshow(display_img)
        axes[0].set_title("Input MRI Scan", fontsize=11, fontweight='bold', color='#1E293B')
        axes[0].axis('off')
        
        axes[1].imshow(cam_np, cmap='jet')
        axes[1].set_title(f"Grad-CAM Heatmap\n(Layer: {target_layer})", fontsize=10.5, fontweight='bold', color='#0F4C81')
        axes[1].axis('off')
        
        axes[2].imshow(overlay_img)
        axes[2].set_title("Overlay (Jet colormap)", fontsize=11, fontweight='bold', color='#1E293B')
        axes[2].axis('off')
        
        # Banner with true vs predicted
        caption = (
            f"Model: {case['model']}  |  True: {case['true_class']}  →  Predicted: {case['predicted_class']}\n"
            f"Confidence: {case['confidence']*100:.1f}%  |  Prediction Margin: {case['margin']*100:.1f}%  |  Normalized Entropy: {case['uncertainty']:.3f}\n"
            f"*Research Note: Highlights regions contributing to the model's prediction; does not establish clinical biomarkers."
        )
        fig.text(0.5, 0.02, caption, ha='center', fontsize=8.5, color='#334155', style='italic')
        
        fname = f"{slug}_error_{idx+1}_{case['true_class'].replace(' ', '_')}_as_{case['predicted_class'].replace(' ', '_')}.png"
        out_path = os.path.join(output_dir, fname)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(out_path, dpi=250, bbox_inches='tight')
        plt.close()
        saved_figures.append(out_path)
        logger.info(f"Saved Grad-CAM error visualization: {out_path}")
        
    return saved_figures


# =====================================================================
# 7. RUN COMPLETE ERROR ANALYSIS PIPELINE
# =====================================================================

def run_complete_error_analysis(
    project_root: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes complete research-grade error analysis on the 960-image test set.
    """
    root = Path(project_root or PROJECT_ROOT)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    logger.info(f"Starting Comprehensive Error Analysis on {device}...")
    
    test_csv = root / "reports" / "splits" / "test.csv"
    val_trans = get_validation_transforms(img_size=(224, 224))
    test_dataset = AlzheimerMRISplitDataset(str(test_csv), project_root=str(root), transform=val_trans)
    
    if len(test_dataset) != 960:
        raise ValueError(f"Expected exactly 960 test samples, found {len(test_dataset)}")
        
    out_dir = root / "results" / "error_analysis"
    fig_dir = root / "results" / "figures" / "error_analysis"
    gradcam_err_dir = fig_dir / "gradcam_errors"
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(gradcam_err_dir, exist_ok=True)
    
    all_model_records: Dict[str, List[Dict[str, Any]]] = {}
    class_summaries: Dict[str, Any] = {}
    conf_summaries: Dict[str, Any] = {}
    
    all_high_conf_errors: List[Dict[str, Any]] = []
    all_low_conf_correct: List[Dict[str, Any]] = []
    
    for display_name, slug, rel_ckpt, rel_temp in MODEL_KEYS:
        ckpt_path = root / rel_ckpt
        temp_path = root / rel_temp
        
        logger.info(f"\n==========================================")
        logger.info(f"Analyzing Errors for {display_name}...")
        logger.info(f"==========================================")
        
        model, msg = load_trained_model(slug, str(ckpt_path), num_classes=4, device=device)
        if model is None:
            raise RuntimeError(f"Failed to load model {display_name}: {msg}")
            
        temp_val = load_temperature(temp_path)
        logger.info(f"Using learned temperature T = {temp_val:.4f} for calibrated probability computation.")
        
        # 1. Extract predictions
        records = extract_model_predictions(model, test_dataset, temperature=temp_val, device=device)
        all_model_records[display_name] = records
        
        # Save prediction CSV
        pred_csv = out_dir / f"{slug}_predictions.csv"
        with open(pred_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        logger.info(f"Saved predictions CSV to {pred_csv}")
        
        # 2. Class-wise error & confusion
        class_summary = compute_class_wise_error_metrics(records)
        class_summaries[display_name] = class_summary
        
        # 3. Confidence & uncertainty stats
        conf_summary = compute_confidence_uncertainty_stats(records)
        conf_summaries[display_name] = conf_summary
        
        # 4. Generate distribution figure
        plot_error_confidence_distribution(
            records=records,
            model_name=display_name,
            save_path=str(fig_dir / f"{slug}_error_confidence.png")
        )
        
        # 5. Extract high-confidence errors (wrong AND conf >= 0.80)
        model_high_errs = []
        for r in records:
            if not r["is_correct"] and r["confidence"] >= 0.80:
                item = {
                    "model": display_name,
                    "architecture_slug": slug,
                    "image_path": r["image_path"],
                    "filename": r["filename"],
                    "true_class": r["true_class"],
                    "predicted_class": r["predicted_class"],
                    "confidence": r["confidence"],
                    "uncertainty": r["entropy_uncertainty"],
                    "margin": r["prediction_margin"]
                }
                all_high_conf_errors.append(item)
                model_high_errs.append(item)
                
        # 6. Extract low-confidence correct (correct AND conf < 0.70)
        for r in records:
            if r["is_correct"] and r["confidence"] < 0.70:
                all_low_conf_correct.append({
                    "model": display_name,
                    "architecture_slug": slug,
                    "image_path": r["image_path"],
                    "filename": r["filename"],
                    "true_class": r["true_class"],
                    "predicted_class": r["predicted_class"],
                    "confidence": r["confidence"],
                    "uncertainty": r["entropy_uncertainty"],
                    "margin": r["prediction_margin"]
                })
                
        # 7. Generate Grad-CAM error visuals for representative high-confidence errors
        if model_high_errs:
            generate_gradcam_for_representative_errors(
                model=model,
                slug=slug,
                high_conf_errors=model_high_errs,
                output_dir=str(gradcam_err_dir),
                max_cases=3
            )
            
    # 8. Save High-Confidence Errors CSV
    high_conf_csv = out_dir / "high_confidence_errors.csv"
    if all_high_conf_errors:
        with open(high_conf_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_high_conf_errors[0].keys()))
            writer.writeheader()
            writer.writerows(all_high_conf_errors)
    else:
        # Create empty with headers
        with open(high_conf_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "architecture_slug", "image_path", "filename", "true_class", "predicted_class", "confidence", "uncertainty", "margin"])
            writer.writeheader()
    logger.info(f"Saved high-confidence errors ({len(all_high_conf_errors)} cases) to {high_conf_csv}")
    
    # 9. Save Low-Confidence Correct CSV
    low_conf_csv = out_dir / "low_confidence_correct.csv"
    if all_low_conf_correct:
        with open(low_conf_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_low_conf_correct[0].keys()))
            writer.writeheader()
            writer.writerows(all_low_conf_correct)
    else:
        with open(low_conf_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "architecture_slug", "image_path", "filename", "true_class", "predicted_class", "confidence", "uncertainty", "margin"])
            writer.writeheader()
    logger.info(f"Saved low-confidence correct ({len(all_low_conf_correct)} cases) to {low_conf_csv}")
    
    # 10. Cross-Model Error Overlap Analysis
    overlap_res = analyze_cross_model_overlap(
        model_predictions=all_model_records,
        save_csv_path=str(out_dir / "cross_model_error_overlap.csv"),
        save_fig_path=str(fig_dir / "cross_model_error_overlap.png")
    )
    
    # 11. Compile and save aggregated summary JSON
    summary_data = {
        "test_sample_count": 960,
        "class_wise_metrics": class_summaries,
        "confidence_uncertainty_stats": conf_summaries,
        "high_confidence_error_counts": {
            "total_ge_0_80": len(all_high_conf_errors),
            "total_ge_0_90": sum(1 for e in all_high_conf_errors if e["confidence"] >= 0.90),
            "total_ge_0_95": sum(1 for e in all_high_conf_errors if e["confidence"] >= 0.95),
            "by_model": {
                m: sum(1 for e in all_high_conf_errors if e["model"] == m) for m in all_model_records.keys()
            }
        },
        "low_confidence_correct_counts": {
            "total_lt_0_70": len(all_low_conf_correct),
            "by_model": {
                m: sum(1 for e in all_low_conf_correct if e["model"] == m) for m in all_model_records.keys()
            }
        },
        "cross_model_overlap": overlap_res
    }
    
    summary_json_path = out_dir / "error_analysis_summary.json"
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)
    logger.info(f"Saved aggregated error analysis summary to {summary_json_path}")
    
    return summary_data


if __name__ == "__main__":
    run_complete_error_analysis()
