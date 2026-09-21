"""
Comprehensive Evaluation Metrics Module for Multi-Stage Alzheimer's MRI Classification.
Calculates Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, Balanced Accuracy, MCC, and Confusion Matrix.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix
)

from src.data.validation import CLASSES, CLASS_DISPLAY_NAMES


def compute_per_class_specificity(cm: np.ndarray) -> List[float]:
    """
    Compute per-class Specificity from multi-class confusion matrix.
    Specificity = TN / (TN + FP)
    """
    num_classes = cm.shape[0]
    specificities = []
    
    total_samples = np.sum(cm)
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fp + fn)
        
        denom = tn + fp
        spec = float(tn / denom) if denom > 0 else 0.0
        specificities.append(spec)
        
    return specificities


def compute_comprehensive_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = CLASSES
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics for multi-class classification.
    
    Args:
        y_true: List or array of ground-truth class indices.
        y_pred: List or array of predicted class indices.
        class_names: List of class names.
        
    Returns:
        Structured dictionary containing all computed metrics.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    num_classes = len(class_names)
    labels = list(range(num_classes))
    
    # 1. Overall Accuracy
    acc = float(accuracy_score(y_true_np, y_pred_np))
    
    # 2. Balanced Accuracy
    bal_acc = float(balanced_accuracy_score(y_true_np, y_pred_np))
    
    # 3. Precision
    prec_macro = float(precision_score(y_true_np, y_pred_np, labels=labels, average='macro', zero_division=0))
    prec_weighted = float(precision_score(y_true_np, y_pred_np, labels=labels, average='weighted', zero_division=0))
    prec_per_class = [float(p) for p in precision_score(y_true_np, y_pred_np, labels=labels, average=None, zero_division=0)]
    
    # 4. Recall / Sensitivity
    rec_macro = float(recall_score(y_true_np, y_pred_np, labels=labels, average='macro', zero_division=0))
    rec_weighted = float(recall_score(y_true_np, y_pred_np, labels=labels, average='weighted', zero_division=0))
    rec_per_class = [float(r) for r in recall_score(y_true_np, y_pred_np, labels=labels, average=None, zero_division=0)]
    
    # 5. F1-Score
    f1_macro_val = float(f1_score(y_true_np, y_pred_np, labels=labels, average='macro', zero_division=0))
    f1_weighted_val = float(f1_score(y_true_np, y_pred_np, labels=labels, average='weighted', zero_division=0))
    f1_per_class = [float(f) for f in f1_score(y_true_np, y_pred_np, labels=labels, average=None, zero_division=0)]
    
    # 6. Confusion Matrix
    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    
    # 7. Specificity
    spec_per_class = compute_per_class_specificity(cm)
    spec_macro = float(np.mean(spec_per_class))
    
    # 8. Matthews Correlation Coefficient (MCC)
    mcc_val = float(matthews_corrcoef(y_true_np, y_pred_np))
    
    # Format per-class breakdown
    per_class_breakdown = {}
    for idx, name in enumerate(class_names):
        per_class_breakdown[name] = {
            "display_name": CLASS_DISPLAY_NAMES.get(name, name),
            "precision": prec_per_class[idx] if idx < len(prec_per_class) else 0.0,
            "recall_sensitivity": rec_per_class[idx] if idx < len(rec_per_class) else 0.0,
            "specificity": spec_per_class[idx] if idx < len(spec_per_class) else 0.0,
            "f1_score": f1_per_class[idx] if idx < len(f1_per_class) else 0.0,
            "support": int(np.sum(y_true_np == idx))
        }
        
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro_sensitivity": rec_macro,
        "recall_weighted": rec_weighted,
        "specificity_macro": spec_macro,
        "f1_macro": f1_macro_val,
        "f1_weighted": f1_weighted_val,
        "matthews_corrcoef": mcc_val,
        "total_test_samples": int(len(y_true_np)),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_breakdown
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASSES,
    title: str = "Confusion Matrix - Alzheimer MRI Classification",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate an annotated seaborn heatmap for the multi-class confusion matrix.
    """
    display_labels = [CLASS_DISPLAY_NAMES.get(c, c).split('(')[0].strip() for c in class_names]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
        ax=ax
    )
    
    ax.set_title(title, fontsize=13, pad=12, fontweight='bold')
    ax.set_ylabel('True Stage', fontsize=11, labelpad=8)
    ax.set_xlabel('Predicted Stage', fontsize=11, labelpad=8)
    plt.xticks(rotation=25, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig
