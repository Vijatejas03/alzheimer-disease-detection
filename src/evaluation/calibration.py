"""
Model Calibration & Reliability Analysis Engine.
Implements Temperature Scaling (Guo et al., 2017), Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), Multi-Class Brier Score, Negative Log-Likelihood (NLL),
and publication-ready Reliability Diagrams and Confidence Distribution plots.

Strict Research Integrity Guarantee:
- Temperature parameter T is optimized STRICTLY on the validation set.
- Evaluation metrics are computed on the untouched held-out test set.
- Pure PyTorch, NumPy, and Matplotlib implementation without external C-extension risks.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import csv
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.model_factory import load_trained_model, MODEL_REGISTRY
from src.data.dataset import create_split_data_loaders
from src.data.validation import CLASSES, CLASS_DISPLAY_NAMES
from src.utils.logger import logger


# =====================================================================
# 1. CORE CALIBRATION METRICS (ECE, MCE, Brier, NLL)
# =====================================================================

def compute_ece_mce(
    probs: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    using equal-width confidence binning.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    MCE = max_{m: |B_m| > 0} |acc(B_m) - conf(B_m)|
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(float)
    
    n_samples = len(targets)
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    
    bin_stats = []
    ece = 0.0
    mce = 0.0
    
    for m in range(num_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        if m == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
        bin_count = int(np.sum(in_bin))
        
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_gap = abs(bin_acc - bin_conf)
            
            ece += (bin_count / n_samples) * bin_gap
            if bin_gap > mce:
                mce = bin_gap
        else:
            bin_acc = 0.0
            bin_conf = float((bin_lower + bin_upper) / 2.0)
            bin_gap = 0.0
            
        bin_stats.append({
            "bin_idx": m,
            "bin_lower": float(bin_lower),
            "bin_upper": float(bin_upper),
            "sample_count": bin_count,
            "accuracy": bin_acc,
            "confidence": bin_conf,
            "gap": float(bin_gap)
        })
        
    return {
        "ece": float(ece),
        "mce": float(mce),
        "num_bins": num_bins,
        "sample_count": n_samples,
        "bin_stats": bin_stats
    }


def compute_brier_score(
    probs: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 4
) -> float:
    """
    Compute multi-class Brier score (strictly proper scoring rule, lower is better):
    Brier = (1 / N) * sum_{i=1}^N sum_{c=1}^C (p_{ic} - y_{ic})^2
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    n_samples = len(targets)
    
    one_hot = np.zeros((n_samples, num_classes), dtype=float)
    one_hot[np.arange(n_samples), targets] = 1.0
    
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)


def compute_nll(
    probs: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-12
) -> float:
    """
    Compute Negative Log-Likelihood (cross-entropy on predicted probabilities):
    NLL = - (1 / N) * sum_{i=1}^N log(p_{i, y_i} + eps)
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    n_samples = len(targets)
    
    p_true = probs[np.arange(n_samples), targets]
    nll = -np.mean(np.log(np.clip(p_true, eps, 1.0)))
    return float(nll)


def compute_comprehensive_calibration_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15,
    num_classes: int = 4
) -> Dict[str, Any]:
    """Calculate all calibration metrics for given probability predictions."""
    ece_mce_res = compute_ece_mce(probs, targets, num_bins=num_bins)
    brier = compute_brier_score(probs, targets, num_classes=num_classes)
    nll = compute_nll(probs, targets)
    
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == targets))
    
    return {
        "accuracy": acc,
        "ece": ece_mce_res["ece"],
        "mce": ece_mce_res["mce"],
        "brier_score": brier,
        "negative_log_likelihood": nll,
        "num_bins": num_bins,
        "sample_count": len(targets),
        "bin_stats": ece_mce_res["bin_stats"]
    }


# =====================================================================
# 2. TEMPERATURE SCALING MODULE (Guo et al., 2017)
# =====================================================================

class TemperatureScaler(nn.Module):
    """
    Post-processing temperature scaling calibration module.
    Scales logits by a single learned scalar parameter T > 0.
    T > 1 softens overconfident probabilities.
    T < 1 sharpens underconfident probabilities.
    Preserves top-1 classification accuracy and ROC rankings identically.
    """
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([float(init_temperature)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> float:
        """
        Fit temperature parameter T strictly on the validation set logits
        by minimizing Negative Log-Likelihood (CrossEntropyLoss).
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()

        def eval_step():
            optimizer.zero_grad()
            # Enforce numerical safety range for temperature
            self.temperature.data.clamp_(min=0.05, max=10.0)
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        self.temperature.data.clamp_(min=0.05, max=10.0)
        
        learned_temp = float(self.temperature.item())
        logger.info(f"Learned optimal temperature T = {learned_temp:.4f}")
        return learned_temp


# =====================================================================
# 3. LOGIT EXTRACTION
# =====================================================================

def extract_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract raw unnormalized logits and ground-truth targets from a DataLoader.
    Runs in pure inference mode without gradient computation.
    """
    model.to(device)
    model.eval()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu())
            all_labels.append(targets.cpu())
            
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    return logits_tensor, labels_tensor


# =====================================================================
# 4. PUBLICATION-QUALITY PLOTTING
# =====================================================================

def plot_reliability_diagram(
    uncal_probs: np.ndarray,
    cal_probs: np.ndarray,
    targets: np.ndarray,
    temperature: float,
    model_name: str,
    save_path: str,
    num_bins: int = 15
) -> None:
    """
    Generate publication-ready side-by-side reliability diagrams
    (Uncalibrated vs. Calibrated) with accuracy bars, confidence gap shading,
    and sample frequency distribution.
    """
    uncal_metrics = compute_comprehensive_calibration_metrics(uncal_probs, targets, num_bins=num_bins)
    cal_metrics = compute_comprehensive_calibration_metrics(cal_probs, targets, num_bins=num_bins)
    
    bin_width = 1.0 / num_bins
    bin_centers = np.linspace(bin_width / 2.0, 1.0 - bin_width / 2.0, num_bins)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    
    # ------------------ Ax1: Uncalibrated ------------------
    ax1.set_facecolor('#F8FAFC')
    uncal_bins = uncal_metrics["bin_stats"]
    accs1 = [b["accuracy"] for b in uncal_bins]
    confs1 = [b["confidence"] for b in uncal_bins]
    counts1 = [b["sample_count"] for b in uncal_bins]
    
    # Diagonal reference
    ax1.plot([0, 1], [0, 1], linestyle='--', color='#64748B', linewidth=1.5, label='Perfect Calibration (y = x)')
    
    # Bars for accuracy
    bars1 = ax1.bar(
        bin_centers, accs1, width=bin_width * 0.9,
        color='#0F4C81', edgecolor='#0A2540', alpha=0.85, label='Empirical Accuracy'
    )
    
    # Gap shading: Red if overconfident (conf > acc), Blue if underconfident
    for i in range(num_bins):
        if counts1[i] > 0:
            if confs1[i] > accs1[i]:
                ax1.bar(
                    bin_centers[i], confs1[i] - accs1[i], bottom=accs1[i],
                    width=bin_width * 0.9, color='#DC2626', alpha=0.45,
                    edgecolor='#DC2626', linestyle=':', label='Overconfidence Gap' if i == 0 else ""
                )
            elif accs1[i] > confs1[i]:
                ax1.bar(
                    bin_centers[i], accs1[i] - confs1[i], bottom=confs1[i],
                    width=bin_width * 0.9, color='#2563EB', alpha=0.35,
                    edgecolor='#2563EB', linestyle=':', label='Underconfidence Gap' if i == 0 else ""
                )
                
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Mean Predicted Confidence', fontsize=11, fontweight='bold', color='#1E293B')
    ax1.set_ylabel('Empirical Accuracy', fontsize=11, fontweight='bold', color='#1E293B')
    ax1.set_title(f'{model_name} — Uncalibrated\n(Original Softmax)', fontsize=12, fontweight='bold', color='#0F4C81')
    ax1.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    
    # Annotation box for metrics
    uncal_text = (
        f"ECE: {uncal_metrics['ece']:.4f} ({uncal_metrics['ece']*100:.2f}%)\n"
        f"MCE: {uncal_metrics['mce']:.4f}\n"
        f"Brier: {uncal_metrics['brier_score']:.4f}\n"
        f"NLL: {uncal_metrics['negative_log_likelihood']:.4f}\n"
        f"Accuracy: {uncal_metrics['accuracy']*100:.2f}%"
    )
    ax1.text(
        0.05, 0.95, uncal_text, transform=ax1.transAxes,
        fontsize=9.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFFF', edgecolor='#CBD5E1', alpha=0.95)
    )
    ax1.legend(loc='lower right', fontsize=8.5)
    
    # ------------------ Ax2: Calibrated ------------------
    ax2.set_facecolor('#F8FAFC')
    cal_bins = cal_metrics["bin_stats"]
    accs2 = [b["accuracy"] for b in cal_bins]
    confs2 = [b["confidence"] for b in cal_bins]
    counts2 = [b["sample_count"] for b in cal_bins]
    
    ax2.plot([0, 1], [0, 1], linestyle='--', color='#64748B', linewidth=1.5, label='Perfect Calibration (y = x)')
    
    bars2 = ax2.bar(
        bin_centers, accs2, width=bin_width * 0.9,
        color='#0D9488', edgecolor='#042F2E', alpha=0.85, label='Empirical Accuracy'
    )
    
    for i in range(num_bins):
        if counts2[i] > 0:
            if confs2[i] > accs2[i]:
                ax2.bar(
                    bin_centers[i], confs2[i] - accs2[i], bottom=accs2[i],
                    width=bin_width * 0.9, color='#DC2626', alpha=0.45,
                    edgecolor='#DC2626', linestyle=':', label='Residual Gap' if i == 0 else ""
                )
            elif accs2[i] > confs2[i]:
                ax2.bar(
                    bin_centers[i], accs2[i] - confs2[i], bottom=confs2[i],
                    width=bin_width * 0.9, color='#2563EB', alpha=0.35,
                    edgecolor='#2563EB', linestyle=':'
                )
                
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Mean Predicted Confidence', fontsize=11, fontweight='bold', color='#1E293B')
    ax2.set_ylabel('Empirical Accuracy', fontsize=11, fontweight='bold', color='#1E293B')
    ax2.set_title(f'{model_name} — Calibrated (T = {temperature:.3f})\n(Temperature Scaled)', fontsize=12, fontweight='bold', color='#0D9488')
    ax2.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    
    cal_text = (
        f"ECE: {cal_metrics['ece']:.4f} ({cal_metrics['ece']*100:.2f}%)\n"
        f"MCE: {cal_metrics['mce']:.4f}\n"
        f"Brier: {cal_metrics['brier_score']:.4f}\n"
        f"NLL: {cal_metrics['negative_log_likelihood']:.4f}\n"
        f"Accuracy: {cal_metrics['accuracy']*100:.2f}%\n"
        f"ECE Reduction: {((uncal_metrics['ece'] - cal_metrics['ece']) / max(1e-6, uncal_metrics['ece']))*100:.1f}%"
    )
    ax2.text(
        0.05, 0.95, cal_text, transform=ax2.transAxes,
        fontsize=9.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFFF', edgecolor='#CBD5E1', alpha=0.95)
    )
    ax2.legend(loc='lower right', fontsize=8.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved reliability diagram to {save_path}")


def plot_confidence_distribution(
    uncal_probs: np.ndarray,
    cal_probs: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    save_path: str
) -> None:
    """
    Generate confidence distribution histograms separating correct and incorrect predictions,
    contrasting uncalibrated vs. calibrated confidence spread.
    """
    targets = np.asarray(targets, dtype=int)
    
    uncal_conf = np.max(uncal_probs, axis=1)
    uncal_pred = np.argmax(uncal_probs, axis=1)
    uncal_correct = (uncal_pred == targets)
    
    cal_conf = np.max(cal_probs, axis=1)
    cal_pred = np.argmax(cal_probs, axis=1)
    cal_correct = (cal_pred == targets)
    
    bins = np.linspace(0.25, 1.0, 31)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    
    # ------------------ Ax1: Uncalibrated ------------------
    ax1.set_facecolor('#F8FAFC')
    ax1.hist(
        uncal_conf[uncal_correct], bins=bins, color='#0F4C81', alpha=0.75,
        edgecolor='#0A2540', label=f'Correct ({np.sum(uncal_correct)})'
    )
    ax1.hist(
        uncal_conf[~uncal_correct], bins=bins, color='#DC2626', alpha=0.75,
        edgecolor='#7F1D1D', label=f'Incorrect ({np.sum(~uncal_correct)})'
    )
    ax1.set_title(f'{model_name} — Uncalibrated Confidence Distribution', fontsize=11.5, fontweight='bold', color='#0F4C81')
    ax1.set_xlabel('Predicted Confidence (max softmax probability)', fontsize=10.5, fontweight='bold', color='#1E293B')
    ax1.set_ylabel('Sample Count', fontsize=10.5, fontweight='bold', color='#1E293B')
    ax1.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax1.legend(loc='upper left', fontsize=9)
    
    # ------------------ Ax2: Calibrated ------------------
    ax2.set_facecolor('#F8FAFC')
    ax2.hist(
        cal_conf[cal_correct], bins=bins, color='#0D9488', alpha=0.75,
        edgecolor='#042F2E', label=f'Correct ({np.sum(cal_correct)})'
    )
    ax2.hist(
        cal_conf[~cal_correct], bins=bins, color='#DC2626', alpha=0.75,
        edgecolor='#7F1D1D', label=f'Incorrect ({np.sum(~cal_correct)})'
    )
    ax2.set_title(f'{model_name} — Calibrated Confidence Distribution', fontsize=11.5, fontweight='bold', color='#0D9488')
    ax2.set_xlabel('Calibrated Confidence (temperature-scaled)', fontsize=10.5, fontweight='bold', color='#1E293B')
    ax2.set_ylabel('Sample Count', fontsize=10.5, fontweight='bold', color='#1E293B')
    ax2.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax2.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confidence distribution plot to {save_path}")


def plot_all_models_calibration_comparison(
    summary_results: Dict[str, Dict[str, Any]],
    save_path: str
) -> None:
    """
    Generate 4-panel multi-model calibration comparison chart for:
    1. Expected Calibration Error (ECE) [lower is better]
    2. Maximum Calibration Error (MCE) [lower is better]
    3. Negative Log-Likelihood (NLL) [lower is better]
    4. Brier Score [lower is better]
    """
    models = list(summary_results.keys())
    x = np.arange(len(models))
    width = 0.35
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    
    metrics_to_plot = [
        ("ece", "Expected Calibration Error (ECE)", axes[0, 0], True),
        ("mce", "Maximum Calibration Error (MCE)", axes[0, 1], True),
        ("negative_log_likelihood", "Negative Log-Likelihood (NLL)", axes[1, 0], False),
        ("brier_score", "Multi-Class Brier Score", axes[1, 1], False)
    ]
    
    for key, title, ax, as_pct in metrics_to_plot:
        ax.set_facecolor('#F8FAFC')
        uncal_vals = [summary_results[m]["uncalibrated"][key] * (100.0 if as_pct else 1.0) for m in models]
        cal_vals = [summary_results[m]["calibrated"][key] * (100.0 if as_pct else 1.0) for m in models]
        
        rects1 = ax.bar(x - width/2, uncal_vals, width, label='Uncalibrated', color='#0F4C81', alpha=0.85, edgecolor='#0A2540')
        rects2 = ax.bar(x + width/2, cal_vals, width, label='Calibrated', color='#0D9488', alpha=0.85, edgecolor='#042F2E')
        
        ax.set_title(title, fontsize=11.5, fontweight='bold', color='#1E293B')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, fontweight='bold', color='#334155')
        ax.set_ylabel('%' if as_pct else 'Loss / Score', fontsize=10, fontweight='bold', color='#475569')
        ax.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
        ax.legend(fontsize=8.5, loc='upper right')
        
        # Value labels on top of bars
        for r in rects1:
            h = r.get_height()
            ax.annotate(f'{h:.2f}' if as_pct else f'{h:.4f}',
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color='#0F4C81', fontweight='bold')
                        
        for r in rects2:
            h = r.get_height()
            ax.annotate(f'{h:.2f}' if as_pct else f'{h:.4f}',
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color='#0D9488', fontweight='bold')
                        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved all-models calibration comparison to {save_path}")


# =====================================================================
# 5. END-TO-END CALIBRATION PIPELINE
# =====================================================================

MODEL_REGISTRY_KEYS = [
    ("MobileNetV2", "mobilenet_v2", "results/models/mobilenet_v2_best.pt"),
    ("EfficientNet-B0", "efficientnet_b0", "results/models/efficientnet_b0_best.pt"),
    ("ResNet18", "resnet18", "results/models/resnet18_best.pt")
]


def run_full_calibration_analysis(
    project_root: str,
    batch_size: int = 32,
    num_bins: int = 15,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes full calibration study across all 3 architectures:
    1. Loads validation loader (960 images) and test loader (960 images).
    2. For each architecture:
       - Extracts validation logits.
       - Fits temperature scaler strictly on validation logits.
       - Saves learned temperature JSON.
       - Extracts test logits from untouched test set.
       - Evaluates uncalibrated test metrics.
       - Evaluates calibrated test metrics.
       - Generates reliability diagram and confidence histogram.
    3. Compiles comparison table (CSV and JSON).
    4. Generates cross-model calibration comparison figure.
    5. Returns complete results dictionary.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    logger.info(f"Starting Full Model Calibration Analysis on {device} (num_bins={num_bins})...")
    
    # Load DataLoaders
    _, val_loader, test_loader, _, _ = create_split_data_loaders(
        splits_dir="reports/splits",
        project_root=project_root,
        batch_size=batch_size,
        pin_memory=(device == 'cuda'),
        num_workers=0
    )
    
    summary_results: Dict[str, Dict[str, Any]] = {}
    csv_rows: List[Dict[str, Any]] = []
    
    cal_dir = os.path.join(project_root, "results", "calibration")
    fig_cal_dir = os.path.join(project_root, "results", "figures", "calibration")
    metrics_dir = os.path.join(project_root, "results", "metrics")
    
    os.makedirs(cal_dir, exist_ok=True)
    os.makedirs(fig_cal_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    for display_name, slug, rel_ckpt_path in MODEL_REGISTRY_KEYS:
        ckpt_path = os.path.join(project_root, rel_ckpt_path)
        logger.info(f"\n==========================================")
        logger.info(f"Calibrating {display_name} ({ckpt_path})...")
        logger.info(f"==========================================")
        
        model, msg = load_trained_model(slug, ckpt_path, num_classes=4, device=device)
        if model is None:
            raise RuntimeError(f"Failed to load model {display_name}: {msg}")
            
        # Step A: Extract validation logits & labels
        logger.info(f"[{display_name}] Extracting validation set logits (960 samples)...")
        val_logits, val_labels = extract_logits_and_labels(model, val_loader, device=device)
        
        # Step B: Fit temperature scaling strictly on validation set
        logger.info(f"[{display_name}] Fitting temperature scaler strictly on validation set...")
        scaler = TemperatureScaler(init_temperature=1.5)
        optimal_temp = scaler.fit(val_logits, val_labels)
        
        # Save temperature JSON
        temp_data = {
            "model_name": display_name,
            "architecture_slug": slug,
            "learned_temperature": optimal_temp,
            "optimization_objective": "Negative Log-Likelihood (CrossEntropyLoss)",
            "optimization_algorithm": "L-BFGS",
            "validation_split_samples": len(val_labels),
            "fitted_on": "reports/splits/validation.csv",
            "test_leakage": False
        }
        temp_save_path = os.path.join(cal_dir, f"{slug}_temperature.json")
        with open(temp_save_path, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, indent=4)
        logger.info(f"Saved learned temperature to {temp_save_path}")
        
        # Step C: Extract untouched test set logits & labels
        logger.info(f"[{display_name}] Extracting held-out test set logits (960 samples)...")
        test_logits, test_labels = extract_logits_and_labels(model, test_loader, device=device)
        
        # Uncalibrated probabilities: softmax(logits)
        uncal_probs = F.softmax(test_logits, dim=1).detach().cpu().numpy()
        
        # Calibrated probabilities: softmax(logits / T)
        with torch.no_grad():
            cal_logits = scaler(test_logits)
            cal_probs = F.softmax(cal_logits, dim=1).detach().cpu().numpy()
        test_labels_np = test_labels.detach().cpu().numpy()
        
        # Step D: Compute metrics
        uncal_metrics = compute_comprehensive_calibration_metrics(uncal_probs, test_labels_np, num_bins=num_bins)
        cal_metrics = compute_comprehensive_calibration_metrics(cal_probs, test_labels_np, num_bins=num_bins)
        
        ece_reduction_pct = ((uncal_metrics["ece"] - cal_metrics["ece"]) / max(1e-6, uncal_metrics["ece"])) * 100.0
        nll_reduction_pct = ((uncal_metrics["negative_log_likelihood"] - cal_metrics["negative_log_likelihood"]) / max(1e-6, uncal_metrics["negative_log_likelihood"])) * 100.0
        brier_reduction_pct = ((uncal_metrics["brier_score"] - cal_metrics["brier_score"]) / max(1e-6, uncal_metrics["brier_score"])) * 100.0
        
        summary_results[display_name] = {
            "model_name": display_name,
            "slug": slug,
            "temperature": optimal_temp,
            "uncalibrated": uncal_metrics,
            "calibrated": cal_metrics,
            "improvements": {
                "ece_reduction_pct": ece_reduction_pct,
                "nll_reduction_pct": nll_reduction_pct,
                "brier_reduction_pct": brier_reduction_pct
            }
        }
        
        csv_rows.append({
            "model_name": display_name,
            "architecture_slug": slug,
            "temperature": f"{optimal_temp:.4f}",
            "uncalibrated_ece": f"{uncal_metrics['ece']:.4f}",
            "calibrated_ece": f"{cal_metrics['ece']:.4f}",
            "ece_reduction_pct": f"{ece_reduction_pct:.2f}%",
            "uncalibrated_mce": f"{uncal_metrics['mce']:.4f}",
            "calibrated_mce": f"{cal_metrics['mce']:.4f}",
            "uncalibrated_brier": f"{uncal_metrics['brier_score']:.4f}",
            "calibrated_brier": f"{cal_metrics['brier_score']:.4f}",
            "uncalibrated_nll": f"{uncal_metrics['negative_log_likelihood']:.4f}",
            "calibrated_nll": f"{cal_metrics['negative_log_likelihood']:.4f}",
            "test_accuracy": f"{cal_metrics['accuracy']:.4f}"
        })
        
        # Step E: Generate diagnostic plots
        rel_fig_path = os.path.join(fig_cal_dir, f"{slug}_reliability.png")
        plot_reliability_diagram(
            uncal_probs=uncal_probs,
            cal_probs=cal_probs,
            targets=test_labels_np,
            temperature=optimal_temp,
            model_name=display_name,
            save_path=rel_fig_path,
            num_bins=num_bins
        )
        # Also copy/save as {slug}_reliability_diagram.png for alternate naming
        import shutil
        alt_rel_path = os.path.join(fig_cal_dir, f"{slug}_reliability_diagram.png")
        shutil.copyfile(rel_fig_path, alt_rel_path)
        
        dist_fig_path = os.path.join(fig_cal_dir, f"{slug}_confidence_histogram.png")
        plot_confidence_distribution(
            uncal_probs=uncal_probs,
            cal_probs=cal_probs,
            targets=test_labels_np,
            model_name=display_name,
            save_path=dist_fig_path
        )
        
    # Step F: Cross-model comparison figure
    comparison_fig_path = os.path.join(fig_cal_dir, "all_models_reliability_comparison.png")
    plot_all_models_calibration_comparison(summary_results, comparison_fig_path)
    
    # Step G: Save CSV and JSON summary
    json_path = os.path.join(metrics_dir, "calibration_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4)
    logger.info(f"Saved calibration JSON results to {json_path}")
    
    csv_path = os.path.join(metrics_dir, "calibration_results.csv")
    fieldnames = [
        "model_name", "architecture_slug", "temperature",
        "uncalibrated_ece", "calibrated_ece", "ece_reduction_pct",
        "uncalibrated_mce", "calibrated_mce",
        "uncalibrated_brier", "calibrated_brier",
        "uncalibrated_nll", "calibrated_nll", "test_accuracy"
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"Saved calibration CSV results to {csv_path}")
    
    # Step H: Save config
    config_path = os.path.join(metrics_dir, "calibration_config.json")
    config_data = {
        "calibration_method": "Temperature Scaling (Guo et al., 2017)",
        "num_bins": num_bins,
        "binning_strategy": "equal_width",
        "optimization_algorithm": "L-BFGS",
        "optimization_loss": "CrossEntropyLoss (Negative Log-Likelihood)",
        "validation_samples": 960,
        "test_samples": 960,
        "validation_split_path": "reports/splits/validation.csv",
        "test_split_path": "reports/splits/test.csv",
        "device": device
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
    logger.info(f"Saved calibration config to {config_path}")
    
    return summary_results


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_full_calibration_analysis(project_root=current_dir)
