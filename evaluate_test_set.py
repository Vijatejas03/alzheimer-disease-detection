"""
Comprehensive Held-Out Test Set Evaluation Engine for Multi-Stage Alzheimer's MRI Detection.
Evaluates MobileNetV2, EfficientNet-B0, and ResNet-18 strictly on the untouched 960-image test set.
Computes Accuracy, Macro Precision/Recall/F1, Per-class Metrics, Specificity, Balanced Accuracy,
MCC, Multiclass ROC-AUC, Confusion Matrices, and generates all required plots and report artifacts.
"""

import os
import sys
import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import logger
from src.data.dataset import AlzheimerMRISplitDataset
from src.data.augmentation import get_validation_transforms
from src.data.validation import CLASSES, CLASS_DISPLAY_NAMES
from src.models.model_factory import build_model, count_parameters, load_trained_model
from src.evaluation.metrics import (
    compute_comprehensive_metrics,
    plot_confusion_matrix,
    plot_multiclass_roc_curve
)


def run_final_test_evaluation():
    logger.info("=" * 80)
    logger.info("  STARTING FINAL HELD-OUT TEST EVALUATION")
    logger.info("  Target Split: 960 Untouched Images strictly quarantined during training")
    logger.info("=" * 80)

    # 1. Verify CUDA Hardware
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available. GPU inference is strictly required.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
    logger.info(f"Verified CUDA Inference Device: {gpu_name} ({vram_mb} MiB VRAM)")

    # 2. Output Directories
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    for d in [METRICS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

    # 3. Data Leakage & Disjointness Verification
    train_csv_path = os.path.join(PROJECT_ROOT, "reports", "splits", "train.csv")
    val_csv_path = os.path.join(PROJECT_ROOT, "reports", "splits", "validation.csv")
    test_csv_path = os.path.join(PROJECT_ROOT, "reports", "splits", "test.csv")

    def read_filepaths(csv_p):
        with open(csv_p, 'r', encoding='utf-8') as f:
            return set(row['filepath'] for row in csv.DictReader(f))

    train_paths = read_filepaths(train_csv_path)
    val_paths = read_filepaths(val_csv_path)
    test_paths = read_filepaths(test_csv_path)

    assert len(test_paths) == 960, f"Expected exactly 960 test samples, found {len(test_paths)}"
    overlap_train_test = train_paths.intersection(test_paths)
    overlap_val_test = val_paths.intersection(test_paths)

    assert len(overlap_train_test) == 0, f"FATAL DATA LEAKAGE: {len(overlap_train_test)} samples overlap between train and test!"
    assert len(overlap_val_test) == 0, f"FATAL DATA LEAKAGE: {len(overlap_val_test)} samples overlap between val and test!"

    logger.info("Data Leakage Audit: PASSED (0% overlap between test set and train/validation sets).")

    # 4. Create Test DataLoader
    test_transform = get_validation_transforms(img_size=(224, 224))
    test_dataset = AlzheimerMRISplitDataset(csv_file=test_csv_path, project_root=PROJECT_ROOT, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    logger.info(f"Loaded held-out test dataset: {len(test_dataset)} samples.")

    # 5. Define Candidate Models & Checkpoints
    candidates = [
        {
            "id": "mobilenet_v2",
            "name": "MobileNetV2",
            "checkpoint": os.path.join(PROJECT_ROOT, "results", "models", "mobilenet_v2_best.pt")
        },
        {
            "id": "efficientnet_b0",
            "name": "EfficientNet-B0",
            "checkpoint": os.path.join(PROJECT_ROOT, "results", "models", "efficientnet_b0_best.pt")
        },
        {
            "id": "resnet18",
            "name": "ResNet-18",
            "checkpoint": os.path.join(PROJECT_ROOT, "results", "models", "resnet18_best.pt")
        }
    ]

    all_test_metrics = {}

    # 6. Evaluation Loop
    for cand in candidates:
        m_id = cand["id"]
        m_name = cand["name"]
        ckpt_path = cand["checkpoint"]

        logger.info(f"\nEvaluating: {m_name} from checkpoint: {ckpt_path}")
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        # Load Model
        model, status = load_trained_model(model_name=m_id, checkpoint_path=ckpt_path, num_classes=4, device=device)
        assert model is not None, f"Failed to load {m_name}: {status}"
        model.eval()

        tot_params, trn_params = count_parameters(model)
        logger.info(f"[{m_name}] Parameters: Total = {tot_params:,} | Trainable = {trn_params:,}")

        all_preds = []
        all_targets = []
        all_probs = []

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        t_start = time.perf_counter()

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(images)
                
                # Health checks
                assert outputs.shape == (images.size(0), 4), f"Unexpected output shape: {outputs.shape}"
                assert not torch.isnan(outputs).any(), f"NaN detected in {m_name} outputs!"
                assert not torch.isinf(outputs).any(), f"Inf detected in {m_name} outputs!"

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        torch.cuda.synchronize()
        eval_time_total = time.perf_counter() - t_start
        peak_eval_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2)
        avg_latency_ms = (eval_time_total / len(all_targets)) * 1000.0

        # Verification of prediction integrity
        assert len(all_targets) == 960, f"Expected 960 evaluated targets, got {len(all_targets)}"
        assert len(all_preds) == 960, f"Expected 960 predictions, got {len(all_preds)}"
        assert set(all_preds).issubset({0, 1, 2, 3}), f"Unexpected prediction classes: {set(all_preds)}"

        # Compute full evaluation metrics
        metrics = compute_comprehensive_metrics(all_targets, all_preds, y_probs=all_probs, class_names=CLASSES)
        metrics["model_id"] = m_id
        metrics["model_name"] = m_name
        metrics["checkpoint_path"] = ckpt_path
        metrics["total_parameters"] = tot_params
        metrics["trainable_parameters"] = trn_params
        metrics["inference_duration_seconds"] = round(eval_time_total, 3)
        metrics["average_latency_ms_per_image"] = round(avg_latency_ms, 2)
        metrics["peak_inference_vram_mb"] = peak_eval_vram_mb
        metrics["total_test_samples_evaluated"] = len(all_targets)

        all_test_metrics[m_id] = metrics

        logger.info(
            f"[{m_name}] Test Acc: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"Balanced Acc: {metrics['balanced_accuracy']:.4f} | "
            f"MCC: {metrics['matthews_corrcoef']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"Latency: {avg_latency_ms:.2f} ms/img"
        )

        # 7. Generate Figures for Each Model
        # Confusion Matrix
        cm_np = np.array(metrics["confusion_matrix"])
        cm_path = os.path.join(FIGURES_DIR, f"{m_id}_test_confusion_matrix.png")
        plot_confusion_matrix(
            cm=cm_np,
            class_names=CLASSES,
            title=f"{m_name} - Test Set Confusion Matrix (N=960)",
            normalize=False,
            save_path=cm_path
        )
        logger.info(f"Saved confusion matrix plot: {cm_path}")

        # ROC Curve
        roc_path = os.path.join(FIGURES_DIR, f"{m_id}_test_roc.png")
        plot_multiclass_roc_curve(
            y_true=np.array(all_targets),
            y_probs=np.array(all_probs),
            class_names=CLASSES,
            model_name=m_name,
            save_path=roc_path
        )
        logger.info(f"Saved ROC curve plot: {roc_path}")

    # 8. Generate Overall Multi-Model Comparison Bar Chart
    comparison_fig_path = os.path.join(FIGURES_DIR, "overall_model_comparison.png")
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ["Accuracy", "Balanced Acc", "Macro Precision", "Macro Recall", "Macro F1", "MCC", "ROC-AUC"]
    x = np.arange(len(metric_names))
    width = 0.25

    model_keys = ["mobilenet_v2", "efficientnet_b0", "resnet18"]
    colors = ["#4a90e2", "#50e3c2", "#f5a623"]

    for idx, m_id in enumerate(model_keys):
        m_data = all_test_metrics[m_id]
        vals = [
            m_data["accuracy"],
            m_data["balanced_accuracy"],
            m_data["macro_precision"],
            m_data["macro_recall_sensitivity"],
            m_data["macro_f1"],
            m_data["matthews_corrcoef"],
            m_data["roc_auc"]
        ]
        rects = ax.bar(x + (idx - 1) * width, vals, width, label=m_data["model_name"], color=colors[idx], edgecolor='black', linewidth=0.6)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Held-Out Test Set Performance Comparison (N=960)', fontsize=13, pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10, fontweight='bold')
    ax.set_ylim([0, 1.08])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(comparison_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved overall comparison plot: {comparison_fig_path}")

    # 9. Save CSV Comparison Table
    csv_path = os.path.join(METRICS_DIR, "final_model_comparison.csv")
    csv_headers = [
        "model_name", "total_parameters", "accuracy", "macro_precision", "macro_recall",
        "macro_f1", "balanced_accuracy", "matthews_corrcoef", "roc_auc", "macro_specificity",
        "non_demented_recall", "very_mild_demented_recall", "mild_demented_recall", "moderate_demented_recall",
        "non_demented_f1", "very_mild_demented_f1", "mild_demented_f1", "moderate_demented_f1",
        "average_latency_ms", "checkpoint_path"
    ]

    rows = []
    for m_id in model_keys:
        d = all_test_metrics[m_id]
        pcm = d["per_class_metrics"]
        rows.append({
            "model_name": d["model_name"],
            "total_parameters": d["total_parameters"],
            "accuracy": round(d["accuracy"], 4),
            "macro_precision": round(d["macro_precision"], 4),
            "macro_recall": round(d["macro_recall_sensitivity"], 4),
            "macro_f1": round(d["macro_f1"], 4),
            "balanced_accuracy": round(d["balanced_accuracy"], 4),
            "matthews_corrcoef": round(d["matthews_corrcoef"], 4),
            "roc_auc": round(d["roc_auc"], 4) if d["roc_auc"] is not None else "N/A",
            "macro_specificity": round(d["macro_specificity"], 4),
            "non_demented_recall": round(pcm["Non_Demented"]["recall_sensitivity"], 4),
            "very_mild_demented_recall": round(pcm["Very_Mild_Demented"]["recall_sensitivity"], 4),
            "mild_demented_recall": round(pcm["Mild_Demented"]["recall_sensitivity"], 4),
            "moderate_demented_recall": round(pcm["Moderate_Demented"]["recall_sensitivity"], 4),
            "non_demented_f1": round(pcm["Non_Demented"]["f1_score"], 4),
            "very_mild_demented_f1": round(pcm["Very_Mild_Demented"]["f1_score"], 4),
            "mild_demented_f1": round(pcm["Mild_Demented"]["f1_score"], 4),
            "moderate_demented_f1": round(pcm["Moderate_Demented"]["f1_score"], 4),
            "average_latency_ms": d["average_latency_ms_per_image"],
            "checkpoint_path": d["checkpoint_path"]
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved CSV model comparison table: {csv_path}")

    # 10. Save JSON Results
    final_json_path = os.path.join(METRICS_DIR, "final_test_results.json")
    json_output = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": "cuda:0",
            "gpu_name": gpu_name,
            "total_vram_mb": vram_mb
        },
        "dataset_split_evaluated": {
            "split_name": "test",
            "total_samples": 960,
            "manifest_file": "reports/splits/test.csv",
            "zero_leakage_verified": True
        },
        "models_evaluated": all_test_metrics,
        "clinical_disclaimer": "This deep learning model is developed strictly for research and academic purposes. It is NOT clinically certified, medically validated, or intended for direct clinical diagnostic or therapeutic use."
    }

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4)
    logger.info(f"Saved final JSON test results: {final_json_path}")

    # 11. Generate Comprehensive Markdown Report
    report_md_path = os.path.join(METRICS_DIR, "final_test_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Held-Out Test Evaluation & Multi-Model Comparative Report\n\n")
        f.write("**Project:** Explainable Deep Learning-Based Multi-Stage Alzheimer's Disease Detection Using Brain MRI  \n")
        f.write(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Hardware Platform:** {gpu_name} ({vram_mb} MiB VRAM)  \n")
        f.write(f"**Held-Out Test Sample Count:** Exactly 960 images (quarantined during all training runs)  \n\n")
        f.write("---\n\n")
        f.write("## 1. Executive Summary Table\n\n")
        f.write("| Model Architecture | Parameters | Accuracy | Macro Precision | Macro Recall | Macro F1 | Balanced Acc | MCC | Multiclass ROC-AUC | Avg Latency |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in rows:
            f.write(f"| **{r['model_name']}** | {r['total_parameters']:,} | **{r['accuracy']*100:.2f}%** | {r['macro_precision']:.4f} | {r['macro_recall']:.4f} | **{r['macro_f1']:.4f}** | {r['balanced_accuracy']:.4f} | {r['matthews_corrcoef']:.4f} | **{r['roc_auc']}** | {r['average_latency_ms']:.2f} ms |\n")
        f.write("\n---\n\n")
        f.write("## 2. Detailed Per-Class Breakdown (Held-Out Test Set)\n\n")
        for m_id in model_keys:
            d = all_test_metrics[m_id]
            pcm = d["per_class_metrics"]
            f.write(f"### {d['model_name']}\n\n")
            f.write("| Class Stage | Support | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC (OvR) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for c_name in CLASSES:
                c_data = pcm[c_name]
                f.write(f"| **{CLASS_DISPLAY_NAMES.get(c_name, c_name)}** | {c_data['support']} | {c_data['precision']:.4f} | {c_data['recall_sensitivity']:.4f} | {c_data['specificity']:.4f} | {c_data['f1_score']:.4f} | {c_data['roc_auc']:.4f} |\n")
            f.write("\n")
        f.write("---\n\n")
        f.write("## 3. Confusion Matrices\n\n")
        for m_id in model_keys:
            d = all_test_metrics[m_id]
            f.write(f"#### {d['model_name']} Confusion Matrix (Row: True, Col: Pred)\n")
            f.write("```\n")
            f.write(np.array2string(np.array(d["confusion_matrix"]), separator=", "))
            f.write("\n```\n\n")
        f.write("---\n\n")
        f.write("## 4. Academic & Research Disclaimer\n\n")
        f.write("> **Notice:** This project is an academic and research prototype developed for multi-stage dementia stage detection from structural MRI. It is NOT clinically certified, medically approved, or intended for independent medical diagnosis.\n")

    logger.info(f"Saved comprehensive test report to: {report_md_path}")
    logger.info("FINAL TEST EVALUATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_final_test_evaluation()
