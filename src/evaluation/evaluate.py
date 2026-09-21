"""
Model Evaluation and Multi-Architecture Benchmarking Engine.
Executes test evaluation passes, generates confusion matrices, and writes unmanipulated metric logs.
"""

import os
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
    
    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader containing test dataset.
        device: 'cpu' or 'cuda'.
        model_name: Model identifier.
        save_results: Whether to save JSON metrics and confusion matrix figure.
        results_dir: Base directory for results.
        
    Returns:
        Dictionary of computed metrics.
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    logger.info(f"Starting test evaluation for {model_name}...")
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
    # Calculate full metrics
    metrics = compute_comprehensive_metrics(all_targets, all_preds, class_names=CLASSES)
    metrics["model_name"] = model_name
    
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
        
        # Save Confusion Matrix Figure
        cm_np = np.array(metrics["confusion_matrix"])
        cm_path = os.path.join(figures_dir, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(cm_np, class_names=CLASSES, title=f"Confusion Matrix ({model_name})", save_path=cm_path)
        logger.info(f"Saved confusion matrix plot to {cm_path}")
        
    return metrics


def load_all_benchmark_results(metrics_dir: str = 'results/metrics') -> Dict[str, Dict[str, Any]]:
    """
    Scan metrics directory and load any recorded test results for candidate models.
    
    Returns:
        Dictionary mapping model names to their recorded metrics.
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
                    m_name = data.get("model_name", os.path.splitext(fname)[0])
                    results[m_name] = data
            except Exception as e:
                logger.error(f"Error reading metrics from {path}: {e}")
                
    return results
