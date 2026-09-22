"""
Standalone Training Runner for MobileNetV2 on NVIDIA GeForce RTX 3050 (4 GB VRAM).
Strictly complies with project governance:
- Dedicated GPU execution (CUDA only, NO CPU fallback)
- Automatic Mixed Precision (AMP / FP16)
- Initial batch size 16 with automatic fallback to 8 and 4 on CUDA OOM
- Strict held-out test set quarantine (test split NOT touched during training)
- Checkpoint integrity validation
"""

import os
import sys
import time
import json
import traceback
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import logger
from src.utils.config import set_global_seed
from src.data.dataset import create_split_data_loaders
from src.training.train import train_model
from src.models.model_factory import build_model, count_parameters


def run_mobilenet_training():
    logger.info("=" * 75)
    logger.info("  STARTING MOBILENET-V2 REAL TRAINING RUN")
    logger.info("  Target Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)")
    logger.info("=" * 75)

    # 1. Strict CUDA Check
    if not torch.cuda.is_available():
        error_msg = "FATAL: CUDA is not available in the PyTorch environment. CPU fallback is strictly prohibited."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
    logger.info(f"Verified CUDA Hardware: {gpu_name} | Total VRAM: {total_vram_mb} MiB")

    # 2. Set Global Seed
    SEED = 42
    set_global_seed(SEED)
    logger.info(f"Global Random Seed set to {SEED}")

    # 3. Hyperparameters
    MODEL_NAME = "mobilenet_v2"
    MAX_EPOCHS = 25
    EARLY_STOPPING_PATIENCE = 7
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    USE_AMP = True
    NUM_WORKERS = 0
    PIN_MEMORY = True
    NUM_CLASSES = 4

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    MODELS_DIR = os.path.join(RESULTS_DIR, "models")
    METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    for d in [MODELS_DIR, METRICS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

    # 4. Batch size selection with OOM fallback loop
    candidate_batch_sizes = [16, 8, 4]
    training_result = None
    actual_batch_size = None
    warnings_or_errors = []

    for bs in candidate_batch_sizes:
        try:
            logger.info(f"\nAttempting training with batch size: {bs}...")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)

            # Load DataLoaders (Held-out test set is quarantined)
            train_loader, val_loader, _, class_weights, train_counts = create_split_data_loaders(
                splits_dir="reports/splits",
                project_root=PROJECT_ROOT,
                batch_size=bs,
                pin_memory=PIN_MEMORY,
                num_workers=NUM_WORKERS
            )

            logger.info(f"Loaded Train Samples: {len(train_loader.dataset):,} | Val Samples: {len(val_loader.dataset):,}")
            logger.info(f"Computed Balanced Class Weights: {class_weights.tolist()}")

            # Re-seed before model initialization
            set_global_seed(SEED)

            training_result = train_model(
                model_name=MODEL_NAME,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=None,  # STRICT: Held-out test set NOT touched during training!
                class_weights=class_weights,
                num_classes=NUM_CLASSES,
                epochs=MAX_EPOCHS,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                patience=EARLY_STOPPING_PATIENCE,
                device="cuda",
                use_amp=USE_AMP,
                results_dir=RESULTS_DIR
            )

            actual_batch_size = bs
            logger.info(f"Training completed successfully with batch size {bs}!")
            break

        except torch.cuda.OutOfMemoryError as oom_err:
            warn_msg = f"CUDA Out Of Memory encountered with batch size {bs}: {oom_err}"
            logger.warning(warn_msg)
            warnings_or_errors.append(warn_msg)
            torch.cuda.empty_cache()
            if bs == candidate_batch_sizes[-1]:
                err_msg = "FATAL: CUDA Out Of Memory persisted even at minimum batch size 4. Halting."
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            logger.info("Retrying with smaller batch size...")
        except Exception as e:
            err_msg = f"FATAL: Unexpected CUDA / training failure: {e}\n{traceback.format_exc()}"
            logger.error(err_msg)
            warnings_or_errors.append(err_msg)
            raise

    # 5. Measure Peak GPU Memory
    peak_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2)
    logger.info(f"Peak GPU VRAM allocated during training run: {peak_vram_mb} MiB / {total_vram_mb} MiB")

    # 6. Checkpoint Integrity Test
    checkpoint_path = training_result["checkpoint_path"]
    logger.info(f"\nRunning Post-Training Checkpoint Integrity Validation on: {checkpoint_path}")
    integrity_result = {}

    try:
        assert os.path.isfile(checkpoint_path), f"Checkpoint file does not exist: {checkpoint_path}"
        checkpoint_data = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)

        required_keys = ['epoch', 'model_name', 'model_state_dict', 'optimizer_state_dict', 'val_macro_f1', 'val_accuracy']
        for k in required_keys:
            assert k in checkpoint_data, f"Missing key in checkpoint: {k}"

        # Instantiate fresh model on CUDA and load weights
        val_model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
        val_model.load_state_dict(checkpoint_data['model_state_dict'])
        val_model.to(device)
        val_model.eval()

        # Run smoke inference on 1 validation batch
        dummy_batch = next(iter(val_loader))[0].to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = val_model(dummy_batch)

        assert logits.shape == (dummy_batch.size(0), NUM_CLASSES), f"Unexpected output shape: {logits.shape}"
        assert not torch.isnan(logits).any(), "NaN values detected in model output logits!"
        assert not torch.isinf(logits).any(), "Infinite values detected in model output logits!"

        integrity_result["status"] = "PASSED"
        integrity_result["checkpoint_exists"] = True
        integrity_result["keys_verified"] = list(checkpoint_data.keys())
        integrity_result["saved_epoch"] = checkpoint_data["epoch"]
        integrity_result["saved_val_macro_f1"] = float(checkpoint_data["val_macro_f1"])
        integrity_result["saved_val_accuracy"] = float(checkpoint_data["val_accuracy"])
        integrity_result["inference_smoke_test_shape"] = list(logits.shape)
        logger.info("==> Checkpoint Integrity & Inference Smoke Test PASSED successfully!")

    except Exception as eval_err:
        integrity_result["status"] = "FAILED"
        integrity_result["error"] = str(eval_err)
        warnings_or_errors.append(f"Checkpoint integrity validation failed: {eval_err}")
        logger.error(f"Checkpoint integrity validation failed: {eval_err}")

    # 7. Write Structured Training Summary
    summary_path = os.path.join(METRICS_DIR, "mobilenet_v2_training_summary.json")
    full_summary = {
        "training_completed_successfully": (training_result is not None and integrity_result.get("status") == "PASSED"),
        "model_architecture": "MobileNetV2",
        "epochs_completed": training_result["epochs_completed"],
        "best_epoch": training_result["best_epoch"],
        "best_val_macro_f1": round(float(training_result["best_val_macro_f1"]), 4),
        "best_val_accuracy": round(float(training_result["best_val_accuracy"]), 4),
        "training_duration_seconds": round(float(training_result["total_training_time_seconds"]), 2),
        "actual_batch_size": actual_batch_size,
        "gpu_name": gpu_name,
        "total_vram_mb": total_vram_mb,
        "peak_vram_mb": peak_vram_mb,
        "amp_enabled": USE_AMP,
        "checkpoint_path": checkpoint_path,
        "history_path": training_result["history_path"],
        "training_curve_path": training_result["curve_path"],
        "checkpoint_integrity_test": integrity_result,
        "warnings_or_errors": warnings_or_errors,
        "clinical_disclaimer": "This deep learning model is developed strictly for research and academic purposes. It is NOT clinically certified, medically validated, or intended for direct clinical diagnostic or therapeutic use."
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=4)
    logger.info(f"Saved complete training summary to: {summary_path}")

    # Output formatted report for the user
    print("\n" + "=" * 70)
    print("  MOBILENET-V2 REAL TRAINING RUN SUMMARY")
    print("=" * 70)
    print(f"1. Training Completed Successfully : {full_summary['training_completed_successfully']}")
    print(f"2. Number of Epochs Completed      : {full_summary['epochs_completed']}")
    print(f"3. Best Epoch                      : {full_summary['best_epoch']}")
    print(f"4. Best Validation Macro F1        : {full_summary['best_val_macro_f1']:.4f}")
    print(f"5. Best Validation Accuracy        : {full_summary['best_val_accuracy']:.4f}")
    print(f"6. Actual Batch Size               : {full_summary['actual_batch_size']}")
    print(f"7. GPU Used                        : {full_summary['gpu_name']}")
    print(f"8. Peak VRAM                       : {full_summary['peak_vram_mb']} MiB / {total_vram_mb} MiB")
    print(f"9. Checkpoint Path                 : {full_summary['checkpoint_path']}")
    print(f"10. Warnings / Errors              : {warnings_or_errors if warnings_or_errors else 'None'}")
    print("=" * 70)

    return full_summary


if __name__ == "__main__":
    run_mobilenet_training()
