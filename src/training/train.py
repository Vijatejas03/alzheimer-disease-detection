"""
Training Pipeline for Candidate Deep Learning Models.
Includes early stopping, class-weighted loss, validation tracking, checkpointing, and metric logging.
"""

import os
import time
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.model_factory import build_model
from src.training.loss import get_loss_function
from src.evaluation.evaluate import evaluate_model_on_test_set
from src.utils.logger import logger


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
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
    device: str = 'cpu'
) -> Dict[str, float]:
    """Evaluate the model on the validation split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def plot_training_curves(history: Dict[str, list], model_name: str, save_path: str):
    """Plot training and validation loss & accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=3)
    ax1.plot(epochs, history['val_loss'], 'r--s', label='Val Loss', markersize=3)
    ax1.set_title(f'{model_name} - Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Accuracy curve
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=3)
    ax2.plot(epochs, history['val_acc'], 'r--s', label='Val Acc', markersize=3)
    ax2.set_title(f'{model_name} - Accuracy History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
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
    device: str = 'cpu',
    results_dir: str = 'results'
) -> Dict[str, Any]:
    """
    Train a candidate architecture with early stopping and checkpoint saving.
    """
    logger.info(f"Initializing training for architecture: {model_name} on {device}")
    
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=True)
    model.to(device)
    
    criterion = get_loss_function(class_weights=class_weights, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    models_dir = os.path.join(results_dir, 'models')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_checkpoint_path = os.path.join(models_dir, f"{model_name}_best.pt")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res = validate_one_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_res["accuracy"])
        
        history['train_loss'].append(train_res['loss'])
        history['train_acc'].append(train_res['accuracy'])
        history['val_loss'].append(val_res['loss'])
        history['val_acc'].append(val_res['accuracy'])
        
        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_res['loss']:.4f}, Train Acc: {train_res['accuracy']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f}, Val Acc: {val_res['accuracy']:.4f}"
        )
        
        # Check for improvement
        if val_res["accuracy"] > best_val_acc:
            best_val_acc = val_res["accuracy"]
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'class_weights': class_weights.cpu() if class_weights is not None else None
            }, best_checkpoint_path)
            logger.info(f"==> Saved new best model checkpoint (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (No improvement for {patience} epochs).")
                break
                
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s. Best Val Acc: {best_val_acc:.4f}")
    
    # Save training curves
    curves_path = os.path.join(figures_dir, f"{model_name}_training_curves.png")
    plot_training_curves(history, model_name, curves_path)
    
    # Load best checkpoint and evaluate on test set if provided
    test_metrics = None
    if test_loader is not None and os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = evaluate_model_on_test_set(
            model=model,
            test_loader=test_loader,
            device=device,
            model_name=model_name,
            results_dir=results_dir
        )
        
    return {
        "model_name": model_name,
        "best_val_accuracy": best_val_acc,
        "total_training_time_seconds": total_time,
        "history": history,
        "checkpoint_path": best_checkpoint_path,
        "test_metrics": test_metrics
    }
