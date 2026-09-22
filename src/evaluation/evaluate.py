"""
Model Evaluation and Multi-Architecture Benchmarking Engine.
Executes test evaluation passes, generates raw & normalized confusion matrices,
measures inference latency, and writes unmanipulated metric logs.
"""

import os
import time
import json
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_comprehensive_metrics, plot_confusion_matrix
from src.data.validation import CLASSES
from src.utils.logger import logger


def evaluate_model_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu',
    model_name: str = 'model',
    save_results: bool = True,
    results_dir: str = 'results'
) -> Dict[str, Any]:
    """
    Perform a complete evaluation pass on a held-out test DataLoader.
    Measures latency, computes multi-class metrics & ROC-AUC, and generates confusion plots.
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    logger.info(f"Starting test evaluation for {model_name} on {device}...")
    
    start_time = time.time()
    total_images_evaluated = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            total_images_evaluated += images.size(0)
            
    total_eval_time = time.time() - start_time
    avg_inference_ms = (total_eval_time / max(1, total_images_evaluated)) * 1000.0
    
    # Calculate full metrics with probabilities
    metrics = compute_comprehensive_metrics(all_targets, all_preds, y_probs=all_probs, class_names=CLASSES)
    metrics["model_name"] = model_name
    metrics["total_inference_time_seconds"] = total_eval_time
    metrics["average_inference_ms"] = avg_inference_ms
    metrics["total_test_samples"] = total_images_evaluated
    
    logger.info(
        f"[{model_name}] Test Acc: {metrics['accuracy']:.4f}, "
        f"Macro F1: {metrics['macro_f1']:.4f}, "
        f"Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
        f"MCC: {metrics['matthews_corrcoef']:.4f}, "
        f"Avg latency: {avg_inference_ms:.2f}ms/image"
    )
    
    if save_results:
        metrics_dir = os.path.join(results_dir, 'metrics')
        figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        # Save JSON
        json_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved evaluation metrics to {json_path}")
        
        cm_np = np.array(metrics["confusion_matrix"])
        
        # 1. Save Raw Confusion Matrix Figure
        cm_path_raw = os.path.join(figures_dir, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(cm_np, class_names=CLASSES, title=f"Confusion Matrix ({model_name})", normalize=False, save_path=cm_path_raw)
        
        # 2. Save Normalized Confusion Matrix Figure
        cm_path_norm = os.path.join(figures_dir, f"{model_name}_confusion_matrix_normalized.png")
        plot_confusion_matrix(cm_np, class_names=CLASSES, title=f"Normalized Confusion Matrix ({model_name})", normalize=True, save_path=cm_path_norm)
        
        logger.info(f"Saved raw & normalized confusion matrix plots to {figures_dir}")
        
    return metrics


def load_all_benchmark_results(metrics_dir: str = 'results/metrics') -> Dict[str, Dict[str, Any]]:
    """
    Scan metrics directory and load any recorded test results for candidate models.
    """
    results = {}
    if not os.path.exists(metrics_dir):
        return results
        
    for fname in os.listdir(metrics_dir):
        if fname.endswith('_metrics.json'):
            path = os.path.join(metrics_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    m_name = data.get("model_name", os.path.splitext(fname)[0].replace('_metrics', ''))
                    results[m_name] = data
            except Exception as e:
                logger.error(f"Error reading metrics from {path}: {e}")
                
    return results
