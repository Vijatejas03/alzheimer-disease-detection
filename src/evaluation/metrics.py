"""
Comprehensive Evaluation Metrics Module for Multi-Stage Alzheimer's MRI Classification.
Calculates Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, Balanced Accuracy,
MCC, Multi-Class ROC-AUC (OvR), and generates raw/normalized Confusion Matrices.
Pure NumPy and Matplotlib implementation - zero external C-extension DLL failure risks.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


from src.data.validation import CLASSES, CLASS_DISPLAY_NAMES


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Compute integer confusion matrix C where C[i, j] is count of true class i predicted as class j."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_roc_auc_ovr(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int = 4) -> Tuple[Optional[float], Optional[List[float]], str]:
    """
    Compute One-vs-Rest ROC-AUC using trapezoidal integration.
    Returns (macro_roc_auc, per_class_roc_auc, status_message).
    """
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    if trapz_fn is None or y_probs is None or len(y_probs) == 0:
        return None, None, "ROC-AUC not reported because predicted probability array is empty or unavailable."
        
    y_probs = np.array(y_probs, dtype=float)
    if y_probs.ndim != 2 or y_probs.shape[1] != num_classes:
        return None, None, f"ROC-AUC not reported because probability shape {y_probs.shape} does not match {num_classes} classes."
        
    aucs = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        n_pos = int(np.sum(y_bin))
        n_neg = len(y_bin) - n_pos
        if n_pos == 0:
            return None, None, f"ROC-AUC not reported because class {c} has 0 positive ground truth samples in evaluation split."
        if n_neg == 0:
            return None, None, f"ROC-AUC not reported because class {c} has 0 negative ground truth samples in evaluation split."
            
        scores = y_probs[:, c]
        desc_order = np.argsort(-scores)
        y_sorted = y_bin[desc_order]
        
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        
        tpr = np.concatenate([[0.0], tps / n_pos])
        fpr = np.concatenate([[0.0], fps / n_neg])
        
        auc_val = float(trapz_fn(tpr, fpr))
        aucs.append(max(0.0, min(1.0, auc_val)))
        
    macro_auc = float(np.mean(aucs))
    return macro_auc, aucs, "Computed successfully"


def compute_comprehensive_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[List[List[float]]] = None,
    class_names: List[str] = CLASSES
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics for multi-class classification.
    """
    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)
    total = len(y_true_np)
    num_classes = len(class_names)
    
    # 1. Confusion Matrix
    cm = compute_confusion_matrix(y_true_np, y_pred_np, num_classes=num_classes)
    
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = total - (tp + fn + fp)
    
    # 2. Per-class metrics
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    
    # Class support (true counts per class)
    support = cm.sum(axis=1)
    
    # Overall Accuracy
    accuracy = float(tp.sum() / total) if total > 0 else 0.0
    
    # Macro metrics (equal weighting across classes)
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    balanced_accuracy = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    macro_specificity = float(np.mean(specificity))
    
    # Weighted metrics (weighted by true sample frequency)
    weights = support.astype(float) / total if total > 0 else np.zeros_like(support, dtype=float)
    weighted_precision = float(np.sum(precision * weights))
    weighted_recall = float(np.sum(recall * weights))
    weighted_f1 = float(np.sum(f1 * weights))
    
    # Matthews Correlation Coefficient (Gorodkin multi-class formulation)
    c = cm.astype(float)
    s = np.sum(c)
    p = np.sum(c, axis=1) # true class sums
    q = np.sum(c, axis=0) # pred class sums
    numerator = np.trace(c) * s - np.dot(p, q)
    denominator = np.sqrt((s**2 - np.dot(p, p)) * (s**2 - np.dot(q, q)))
    mcc_val = float(numerator / denominator) if denominator > 0 else 0.0
    
    # Multi-Class ROC-AUC
    macro_auc, per_class_auc, auc_status = None, None, "No probabilities provided"
    if y_probs is not None:
        macro_auc, per_class_auc, auc_status = compute_roc_auc_ovr(y_true_np, y_probs, num_classes=num_classes)
        
    # Structured per-class breakdown
    per_class_breakdown = {}
    for idx, name in enumerate(class_names):
        per_class_breakdown[name] = {
            "display_name": CLASS_DISPLAY_NAMES.get(name, name),
            "precision": float(precision[idx]),
            "recall_sensitivity": float(recall[idx]),
            "specificity": float(specificity[idx]),
            "f1_score": float(f1[idx]),
            "roc_auc": float(per_class_auc[idx]) if per_class_auc is not None else None,
            "support": int(support[idx])
        }
        
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "precision_weighted": weighted_precision,
        "macro_recall_sensitivity": macro_recall,
        "recall_weighted": weighted_recall,
        "macro_specificity": macro_specificity,
        "macro_f1": macro_f1,
        "f1_weighted": weighted_f1,
        "matthews_corrcoef": mcc_val,
        "roc_auc": macro_auc,
        "roc_auc_status": auc_status,
        "total_test_samples": int(total),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_breakdown
    }


compute_all_metrics = compute_comprehensive_metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASSES,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate an annotated heatmap for the multi-class confusion matrix using pure Matplotlib.
    Supports both raw counts and normalized row percentages.
    """
    display_labels = [CLASS_DISPLAY_NAMES.get(c, c).split('(')[0].strip() for c in class_names]
    num_classes = len(class_names)
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        plot_data = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        cbar_label = 'Normalized Proportion'
    else:
        plot_data = cm
        cbar_label = 'Sample Count'
        
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(plot_data, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=11, labelpad=8)
    
    # Tick marks
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(display_labels, rotation=25, ha='right', fontsize=10)
    ax.set_yticklabels(display_labels, fontsize=10)
    
    # Text annotations
    thresh = plot_data.max() / 2.0 if plot_data.max() > 0 else 1.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = plot_data[i, j]
            text_str = f"{val:.2%}" if normalize else f"{int(val)}"
            ax.text(
                j, i, text_str,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11, fontweight="bold"
            )
            
    ax.set_title(title, fontsize=13, pad=12, fontweight='bold')
    ax.set_ylabel('True Stage', fontsize=11, labelpad=8)
    ax.set_xlabel('Predicted Stage', fontsize=11, labelpad=8)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_multiclass_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str] = CLASSES,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot One-vs-Rest (OvR) ROC curves for each class along with the macro-average ROC curve.
    """
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    if trapz_fn is None or y_probs is None or len(y_probs) == 0:
        return None
        
    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs, dtype=float)
    num_classes = len(class_names)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)
    valid_classes = 0
    
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        n_pos = int(np.sum(y_bin))
        n_neg = len(y_bin) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
            
        scores = y_probs[:, c]
        desc_order = np.argsort(-scores)
        y_sorted = y_bin[desc_order]
        
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        
        tpr = np.concatenate([[0.0], tps / n_pos])
        fpr = np.concatenate([[0.0], fps / n_neg])
        
        auc_val = float(trapz_fn(tpr, fpr))
        auc_val = max(0.0, min(1.0, auc_val))
        
        # Interpolate for macro curve
        interp_tpr = np.interp(all_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        mean_tpr += interp_tpr
        valid_classes += 1
        
        c_label = CLASS_DISPLAY_NAMES.get(class_names[c], class_names[c]).split('(')[0].strip()
        color = colors[c % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{c_label} (AUC = {auc_val:.4f})')
        
    if valid_classes > 0:
        mean_tpr /= valid_classes
        mean_tpr[-1] = 1.0
        macro_auc = float(trapz_fn(mean_tpr, all_fpr))
        ax.plot(all_fpr, mean_tpr, color='navy', linestyle='--', lw=2.5, label=f'Macro-Average ROC (AUC = {macro_auc:.4f})')
        
    ax.plot([0, 1], [0, 1], 'k:', lw=1.5, label='Random Chance (AUC = 0.5000)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, labelpad=8)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, labelpad=8)
    ax.set_title(f'{model_name} - Multiclass ROC Curves (One-vs-Rest)', fontsize=12, pad=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

