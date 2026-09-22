"""
Controlled Robustness and Perturbation Evaluation Module for Multi-Stage Alzheimer's MRI.
Evaluates model stability under mild, realistic scanner-like variations:
- Brightness shifts (+/-15%)
- Contrast variations (+/-15%)
- Mild Gaussian noise (sigma=0.03)
- Mild Gaussian blur (radius=0.75)
- Small head tilt rotations (+/-5 degrees)
- Mild scale/zoom variations (0.92x and 1.08x)

Strict Research Integrity Guarantee:
- Official held-out test split metrics remain untouched.
- Robustness evaluation is executed in a separate derived experiment.
- Pure PyTorch, PIL, NumPy, and Matplotlib implementation without external C-extension risks.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.model_factory import load_trained_model, MODEL_REGISTRY
from src.data.dataset import AlzheimerMRISplitDataset
from src.data.augmentation import get_validation_transforms
from src.data.validation import CLASSES
from src.evaluation.metrics import compute_comprehensive_metrics
from src.utils.logger import logger


# =====================================================================
# 1. PERTURBATION DEFINITIONS
# =====================================================================

def apply_brightness_low(img: Image.Image) -> Image.Image:
    """Brightness decreased by 15% (factor 0.85)."""
    return ImageEnhance.Brightness(img).enhance(0.85)


def apply_brightness_high(img: Image.Image) -> Image.Image:
    """Brightness increased by 15% (factor 1.15)."""
    return ImageEnhance.Brightness(img).enhance(1.15)


def apply_contrast_low(img: Image.Image) -> Image.Image:
    """Contrast decreased by 15% (factor 0.85)."""
    return ImageEnhance.Contrast(img).enhance(0.85)


def apply_contrast_high(img: Image.Image) -> Image.Image:
    """Contrast increased by 15% (factor 1.15)."""
    return ImageEnhance.Contrast(img).enhance(1.15)


def apply_gaussian_noise(img: Image.Image, sigma: float = 0.03, seed: Optional[int] = None) -> Image.Image:
    """Additive zero-mean Gaussian noise (sigma=0.03 in [0.0, 1.0] image space, clipped to [0, 255])."""
    arr = np.array(img, dtype=np.float32) / 255.0
    rng = np.random.RandomState(seed) if seed is not None else np.random
    noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy_arr = np.clip(arr + noise, 0.0, 1.0) * 255.0
    return Image.fromarray(noisy_arr.astype(np.uint8))


def apply_gaussian_blur(img: Image.Image, radius: float = 0.75) -> Image.Image:
    """Mild Gaussian blur (radius=0.75, simulating slight PSF / micro-motion)."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_rotation_pos(img: Image.Image) -> Image.Image:
    """Small head tilt +5 degrees (bilinear interpolation)."""
    return img.rotate(5, resample=Image.Resampling.BILINEAR, fillcolor=0)


def apply_rotation_neg(img: Image.Image) -> Image.Image:
    """Small head tilt -5 degrees (bilinear interpolation)."""
    return img.rotate(-5, resample=Image.Resampling.BILINEAR, fillcolor=0)


def apply_scale_down(img: Image.Image) -> Image.Image:
    """Mild zoom-out (0.92x scale, padded with black borders to 224x224)."""
    orig_w, orig_h = img.size
    new_w, new_h = int(orig_w * 0.92), int(orig_h * 0.92)
    scaled = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (orig_w, orig_h), color=0)
    canvas.paste(scaled, ((orig_w - new_w) // 2, (orig_h - new_h) // 2))
    return canvas


def apply_scale_up(img: Image.Image) -> Image.Image:
    """Mild zoom-in (1.08x scale, center-cropped to 224x224)."""
    orig_w, orig_h = img.size
    new_w, new_h = int(orig_w * 1.08), int(orig_h * 1.08)
    scaled = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - orig_w) // 2
    top = (new_h - orig_h) // 2
    return scaled.crop((left, top, left + orig_w, top + orig_h))


PERTURBATION_SUITE = [
    {
        "key": "brightness_minus_15",
        "category": "brightness",
        "label": "Brightness (-15%)",
        "description": "Simulates subtle underexposure / RF receive gain attenuation",
        "fn": apply_brightness_low
    },
    {
        "key": "brightness_plus_15",
        "category": "brightness",
        "label": "Brightness (+15%)",
        "description": "Simulates subtle overexposure / RF gain increase",
        "fn": apply_brightness_high
    },
    {
        "key": "contrast_minus_15",
        "category": "contrast",
        "label": "Contrast (-15%)",
        "description": "Simulates lowered gray-matter / white-matter differentiation",
        "fn": apply_contrast_low
    },
    {
        "key": "contrast_plus_15",
        "category": "contrast",
        "label": "Contrast (+15%)",
        "description": "Simulates enhanced tissue contrast",
        "fn": apply_contrast_high
    },
    {
        "key": "gaussian_noise_003",
        "category": "noise",
        "label": "Gaussian Noise (sigma=0.03)",
        "description": "Simulates mild thermal / RF receiver coil noise",
        "fn": lambda img, seed=None: apply_gaussian_noise(img, sigma=0.03, seed=seed)
    },
    {
        "key": "gaussian_blur_075",
        "category": "blur",
        "label": "Gaussian Blur (r=0.75)",
        "description": "Simulates subtle scanner point-spread function broadening or micro-motion",
        "fn": lambda img, seed=None: apply_gaussian_blur(img, radius=0.75)
    },
    {
        "key": "rotation_minus_5",
        "category": "rotation",
        "label": "Rotation (-5 deg)",
        "description": "Simulates minor head tilt to the right",
        "fn": lambda img, seed=None: apply_rotation_neg(img)
    },
    {
        "key": "rotation_plus_5",
        "category": "rotation",
        "label": "Rotation (+5 deg)",
        "description": "Simulates minor head tilt to the left",
        "fn": lambda img, seed=None: apply_rotation_pos(img)
    },
    {
        "key": "scale_down_92",
        "category": "scale",
        "label": "Scale (0.92x)",
        "description": "Simulates mild zoom-out / FOV enlargement",
        "fn": lambda img, seed=None: apply_scale_down(img)
    },
    {
        "key": "scale_up_108",
        "category": "scale",
        "label": "Scale (1.08x)",
        "description": "Simulates mild zoom-in / brain enlargement",
        "fn": lambda img, seed=None: apply_scale_up(img)
    }
]


# =====================================================================
# 2. PERTURBED DATASET WRAPPER
# =====================================================================

class PerturbedAlzheimerDataset(Dataset):
    """
    Applies a specified deterministic perturbation function to PIL images
    prior to standard validation tensor normalization.
    """
    def __init__(
        self,
        base_dataset: AlzheimerMRISplitDataset,
        perturbation_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
        tensor_transform = None
    ):
        self.base_dataset = base_dataset
        self.perturbation_fn = perturbation_fn
        self.tensor_transform = tensor_transform or get_validation_transforms(img_size=(224, 224))
        
    def __len__(self) -> int:
        return len(self.base_dataset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.base_dataset.samples[idx]
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            pil_img = Image.new('RGB', (224, 224), color=0)
            
        if self.perturbation_fn is not None:
            try:
                pil_img = self.perturbation_fn(pil_img, seed=42 + idx)
            except TypeError:
                pil_img = self.perturbation_fn(pil_img)
            
        tensor = self.tensor_transform(pil_img)
        return tensor, label


# =====================================================================
# 3. ROBUSTNESS EVALUATION ROUTINE
# =====================================================================

MODEL_CONFIGS = [
    ("MobileNetV2", "mobilenet_v2", "results/models/mobilenet_v2_best.pt", 0.93125),
    ("EfficientNet-B0", "efficientnet_b0", "results/models/efficientnet_b0_best.pt", 0.98229),
    ("ResNet18", "resnet18", "results/models/resnet18_best.pt", 0.98438)
]


def evaluate_perturbation(
    model: nn.Module,
    loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate accuracy and Macro F1 on a DataLoader."""
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())
            
    metrics = compute_comprehensive_metrics(all_targets, all_preds, class_names=CLASSES)
    return float(metrics["accuracy"]), float(metrics["macro_f1"]), all_preds, all_targets


# =====================================================================
# 4. ROBUSTNESS PLOTTING
# =====================================================================

def plot_model_robustness_comparison(
    results_by_model: Dict[str, List[Dict[str, Any]]],
    save_path: str
) -> None:
    """
    Generate grouped bar chart showing accuracy under all perturbations for all 3 models.
    """
    models = list(results_by_model.keys())
    pert_labels = [r["label"] for r in results_by_model[models[0]]]
    
    x = np.arange(len(pert_labels))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    palette = ['#2563EB', '#0D9488', '#D97706']
    
    for i, m in enumerate(models):
        accs = [r["perturbed_accuracy"] * 100.0 for r in results_by_model[m]]
        offset = (i - 1) * width
        rects = ax.bar(x + offset, accs, width, label=m, color=palette[i], alpha=0.88, edgecolor='#0F172A')
        
    ax.set_title("Model Robustness Under Controlled Perturbations on Held-Out Test Set (N = 960)", fontsize=13, fontweight='bold', color='#0F4C81')
    ax.set_xticks(x)
    ax.set_xticklabels(pert_labels, rotation=30, ha='right', fontsize=9.5, fontweight='bold', color='#1E293B')
    ax.set_ylabel("Perturbed Accuracy (%)", fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_ylim([75.0, 101.0])
    ax.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax.legend(fontsize=10, loc='lower left')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model robustness comparison figure to {save_path}")


def plot_category_robustness(
    results_by_model: Dict[str, List[Dict[str, Any]]],
    category: str,
    title: str,
    save_path: str
) -> None:
    """Generate bar chart for a specific perturbation category (brightness, contrast, noise, blur, rotation)."""
    models = list(results_by_model.keys())
    
    # Filter by category
    sample_m = models[0]
    cat_items = [r for r in results_by_model[sample_m] if r["category"] == category]
    labels = ["Clean Test Set"] + [r["label"] for r in cat_items]
    
    x = np.arange(len(labels))
    width = 0.25
    palette = ['#2563EB', '#0D9488', '#D97706']
    
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    for i, m in enumerate(models):
        baseline_acc = results_by_model[m][0]["baseline_accuracy"] * 100.0
        pert_accs = [r["perturbed_accuracy"] * 100.0 for r in results_by_model[m] if r["category"] == category]
        vals = [baseline_acc] + pert_accs
        
        offset = (i - 1) * width
        rects = ax.bar(x + offset, vals, width, label=m, color=palette[i], alpha=0.88, edgecolor='#0F172A')
        for r in rects:
            h = r.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color='#1E293B', fontweight='bold')
            
    ax.set_title(title, fontsize=12, fontweight='bold', color='#0F4C81')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold', color='#1E293B')
    ax.set_ylabel("Accuracy (%)", fontsize=10.5, fontweight='bold', color='#1E293B')
    ax.set_ylim([75.0, 102.0])
    ax.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax.legend(fontsize=9, loc='lower left')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved category robustness plot to {save_path}")


# =====================================================================
# 5. MAIN ROBUSTNESS PIPELINE
# =====================================================================

def run_controlled_robustness_experiments(
    project_root: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes full robustness study across MobileNetV2, EfficientNet-B0, and ResNet18.
    """
    root = Path(project_root or PROJECT_ROOT)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    logger.info(f"Starting Controlled Robustness Evaluation on {device}...")
    
    test_csv = root / "reports" / "splits" / "test.csv"
    base_test_dataset = AlzheimerMRISplitDataset(str(test_csv), project_root=str(root), transform=None)
    
    val_trans = get_validation_transforms(img_size=(224, 224))
    
    out_dir = root / "results" / "robustness"
    fig_dir = root / "results" / "figures" / "robustness"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    results_by_model: Dict[str, List[Dict[str, Any]]] = {}
    csv_rows: List[Dict[str, Any]] = []
    
    # Pre-build DataLoaders for each perturbation to avoid duplicate memory allocation
    logger.info("Pre-building Perturbed DataLoaders...")
    loaders = {}
    for p in PERTURBATION_SUITE:
        pert_ds = PerturbedAlzheimerDataset(
            base_dataset=base_test_dataset,
            perturbation_fn=p["fn"],
            tensor_transform=val_trans
        )
        loaders[p["key"]] = DataLoader(pert_ds, batch_size=32, shuffle=False, num_workers=0)
        
    for display_name, slug, rel_ckpt, baseline_acc in MODEL_CONFIGS:
        ckpt_path = root / rel_ckpt
        logger.info(f"\n==========================================")
        logger.info(f"Evaluating Robustness for {display_name}...")
        logger.info(f"==========================================")
        
        model, msg = load_trained_model(slug, str(ckpt_path), num_classes=4, device=device)
        if model is None:
            raise RuntimeError(f"Failed to load model {display_name}: {msg}")
            
        model_results = []
        for p in PERTURBATION_SUITE:
            loader = loaders[p["key"]]
            pert_acc, pert_f1, _, _ = evaluate_perturbation(model, loader, device=device)
            acc_change = float(pert_acc - baseline_acc)
            rel_drop_pct = float((acc_change / max(1e-4, baseline_acc)) * 100.0)
            
            logger.info(f"[{display_name}] {p['label']:<24}: Acc = {pert_acc*100:.2f}% (Delta = {acc_change*100:+.2f}%), Macro F1 = {pert_f1:.4f}")
            
            res_item = {
                "model_name": display_name,
                "architecture_slug": slug,
                "perturbation_key": p["key"],
                "category": p["category"],
                "label": p["label"],
                "description": p["description"],
                "baseline_accuracy": float(baseline_acc),
                "perturbed_accuracy": pert_acc,
                "accuracy_change": acc_change,
                "relative_drop_pct": rel_drop_pct,
                "perturbed_macro_f1": pert_f1
            }
            model_results.append(res_item)
            csv_rows.append(res_item)
            
        results_by_model[display_name] = model_results
        
    # Save CSV
    csv_path = out_dir / "robustness_results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"Saved robustness CSV results to {csv_path}")
    
    # Save JSON
    json_path = out_dir / "robustness_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_by_model, f, indent=4)
    logger.info(f"Saved robustness JSON results to {json_path}")
    
    # Generate Figures
    logger.info("Generating robustness diagnostic figures...")
    plot_model_robustness_comparison(results_by_model, str(fig_dir / "model_robustness_comparison.png"))
    plot_category_robustness(results_by_model, "brightness", "Robustness Under Brightness Variations (+/-15%)", str(fig_dir / "brightness_robustness.png"))
    plot_category_robustness(results_by_model, "contrast", "Robustness Under Contrast Variations (+/-15%)", str(fig_dir / "contrast_robustness.png"))
    plot_category_robustness(results_by_model, "noise", "Robustness Under Additive Gaussian Noise (σ=0.03)", str(fig_dir / "noise_robustness.png"))
    plot_category_robustness(results_by_model, "blur", "Robustness Under Gaussian Blur (r=0.75)", str(fig_dir / "blur_robustness.png"))
    plot_category_robustness(results_by_model, "rotation", "Robustness Under Small Head Tilts (+/-5°)", str(fig_dir / "rotation_robustness.png"))
    
    return results_by_model


if __name__ == "__main__":
    run_controlled_robustness_experiments()
