"""
Training Pipeline for Candidate Deep Learning Models.
Includes early stopping based on Macro F1, class-weighted loss, validation tracking,
checkpointing, metric logging, training curve visualization, and Automatic Mixed Precision (AMP).
"""

import os
import time
import json
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.model_factory import build_model
from src.training.loss import get_loss_function
from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.evaluate import evaluate_model_on_test_set
from src.utils.logger import logger


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = 'cpu',
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False
) -> Dict[str, float]:
    """Train the model for one epoch with optional Automatic Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    device_type = 'cuda' if 'cuda' in str(device).lower() else 'cpu'
    
    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and device_type == 'cuda' and scaler is not None:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = 'cpu',
    use_amp: bool = False
) -> Dict[str, Any]:
    """Evaluate the model on the validation split and calculate multi-class metrics."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    device_type = 'cuda' if 'cuda' in str(device).lower() else 'cpu'
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if use_amp and device_type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            running_loss += loss.item() * images.size(0)
            total += targets.size(0)
            
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
    epoch_loss = running_loss / max(1, total)
    val_metrics = compute_comprehensive_metrics(all_targets, all_preds, y_probs=all_probs)
    val_metrics["loss"] = epoch_loss
    return val_metrics


def plot_training_curves(history: Dict[str, list], model_name: str, save_path: str):
    """Plot training and validation loss, accuracy, and Macro F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=3)
    ax1.plot(epochs, history['val_loss'], 'r--s', label='Val Loss', markersize=3)
    ax1.set_title(f'{model_name} - Loss History', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Accuracy and Macro F1 curves
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=3)
    ax2.plot(epochs, history['val_acc'], 'r--s', label='Val Acc', markersize=3)
    ax2.plot(epochs, history['val_macro_f1'], 'g-.^', label='Val Macro F1', markersize=3)
    ax2.set_title(f'{model_name} - Accuracy & Macro F1', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    class_weights: Optional[torch.Tensor] = None,
    num_classes: int = 4,
    epochs: int = 25,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    device: str = 'auto',
    use_amp: bool = True,
    results_dir: str = 'results'
) -> Dict[str, Any]:
    """
    Train a candidate architecture with early stopping based on Macro F1 and checkpoint saving.
    Supports CUDA with Automatic Mixed Precision (AMP) and CPU fallback.
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    device_type = 'cuda' if 'cuda' in str(device).lower() else 'cpu'
    actual_use_amp = (device_type == 'cuda' and torch.cuda.is_available()) and use_amp
    scaler = torch.amp.GradScaler('cuda', enabled=actual_use_amp) if actual_use_amp else None
    
    logger.info(f"Initializing training for architecture: {model_name} on {device} (AMP: {actual_use_amp})")
    
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=True)
    model.to(device)
    
    criterion = get_loss_function(class_weights=class_weights, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    models_dir = os.path.join(results_dir, 'models')
    figures_dir = os.path.join(results_dir, 'figures')
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    best_val_macro_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = os.path.join(models_dir, f"{model_name}_best.pt")
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_macro_f1': [],
        'val_balanced_acc': [],
        'val_mcc': [],
        'learning_rate': [],
        'epoch_duration_seconds': []
    }
    
    start_total_time = time.time()
    
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        
        train_res = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=actual_use_amp
        )
        val_res = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=actual_use_amp
        )
        
        epoch_duration = time.time() - t_epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step LR scheduler on validation Macro F1
        scheduler.step(val_res["macro_f1"])
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_res['loss'])
        history['train_acc'].append(train_res['accuracy'])
        history['val_loss'].append(val_res['loss'])
        history['val_acc'].append(val_res['accuracy'])
        history['val_macro_f1'].append(val_res['macro_f1'])
        history['val_balanced_acc'].append(val_res['balanced_accuracy'])
        history['val_mcc'].append(val_res['matthews_corrcoef'])
        history['learning_rate'].append(current_lr)
        history['epoch_duration_seconds'].append(epoch_duration)
        
        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) "
            f"Train Loss: {train_res['loss']:.4f}, Train Acc: {train_res['accuracy']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f}, Val Acc: {val_res['accuracy']:.4f}, "
            f"Val Macro F1: {val_res['macro_f1']:.4f}, Bal Acc: {val_res['balanced_accuracy']:.4f}, MCC: {val_res['matthews_corrcoef']:.4f}"
        )
        
        # Check for improvement based on Validation Macro F1
        if val_res["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_res["macro_f1"]
            best_val_acc = val_res["accuracy"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': best_val_macro_f1,
                'val_accuracy': best_val_acc,
                'class_weights': class_weights.cpu() if class_weights is not None else None
            }, best_checkpoint_path)
            logger.info(f"==> Saved new best model checkpoint (Epoch {epoch}, Val Macro F1: {best_val_macro_f1:.4f}, Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (No Macro F1 improvement for {patience} epochs).")
                break
                
    total_training_time = time.time() - start_total_time
    logger.info(f"Training finished for {model_name} in {total_training_time:.1f}s. Best Epoch: {best_epoch}, Best Val Macro F1: {best_val_macro_f1:.4f}")
    
    # Save training curves
    curve_path = os.path.join(figures_dir, f"{model_name}_training_curve.png")
    plot_training_curves(history, model_name, curve_path)
    logger.info(f"Saved training curve to {curve_path}")
    
    # Save training history JSON
    history_path = os.path.join(metrics_dir, f"{model_name}_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved training history to {history_path}")
    
    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_acc,
        "epochs_completed": len(history['epoch']),
        "total_training_time_seconds": total_training_time,
        "checkpoint_path": best_checkpoint_path,
        "history_path": history_path,
        "curve_path": curve_path
    }
