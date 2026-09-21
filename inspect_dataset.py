"""
Dataset Inspection Utility
===========================
Standalone script to inspect an Alzheimer MRI dataset directory before model training.

Reports:
  - Number of images per class and total
  - Image dimension statistics (min, max, mean)
  - File format distribution
  - Corrupted / unreadable images
  - Duplicate filenames (within or across classes)
  - Approximate duplicate image content via perceptual hash
  - Class distribution and imbalance ratio
  - Recommended train / val / test split sizes

Usage:
  python inspect_dataset.py
  python inspect_dataset.py --dataset_dir path/to/dataset
  python inspect_dataset.py --dataset_dir path/to/dataset --check_duplicates
"""

import os
import sys
import argparse
import hashlib
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

# ── Class configuration ───────────────────────────────────────────────────────
EXPECTED_CLASSES = [
    'Non_Demented',
    'Very_Mild_Demented',
    'Mild_Demented',
    'Moderate_Demented',
]

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# ── Helpers ───────────────────────────────────────────────────────────────────

def file_md5(path: str) -> Optional[str]:
    """Compute MD5 hash of a file for exact duplicate detection."""
    try:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def average_hash(img: Image.Image, hash_size: int = 8) -> Optional[str]:
    """
    Compute a simple perceptual (average) hash for near-duplicate detection.
    Resizes image to hash_size x hash_size greyscale and thresholds at mean.
    """
    try:
        small = img.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(small.getdata())
        mean_val = sum(pixels) / len(pixels)
        bits = ''.join('1' if p >= mean_val else '0' for p in pixels)
        return bits
    except Exception:
        return None


def hamming_distance(h1: str, h2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


# ── Main inspection ───────────────────────────────────────────────────────────

def inspect_dataset(
    dataset_dir: str,
    check_perceptual_duplicates: bool = False,
    phash_threshold: int = 5
) -> Dict:
    """
    Perform a comprehensive inspection of the dataset directory.

    Args:
        dataset_dir: Path to dataset root with class subdirectories.
        check_perceptual_duplicates: If True, compute perceptual hashes (slow for large datasets).
        phash_threshold: Hamming distance threshold to flag near-duplicate pairs.

    Returns:
        Dictionary containing the full inspection report.
    """
    print(f"\n{'='*70}")
    print(f"  ALZHEIMER MRI DATASET INSPECTION REPORT")
    print(f"{'='*70}")
    print(f"  Dataset directory : {os.path.abspath(dataset_dir)}")
    print(f"  Expected classes  : {EXPECTED_CLASSES}")
    print()

    report = {
        "dataset_dir": os.path.abspath(dataset_dir),
        "expected_classes": EXPECTED_CLASSES,
        "classes_found": [],
        "classes_missing": [],
        "per_class": {},
        "total_valid_images": 0,
        "total_corrupted": 0,
        "format_distribution": defaultdict(int),
        "dimension_stats": {},
        "duplicate_filenames": [],
        "exact_duplicate_files": [],
        "perceptual_duplicate_pairs": [],
        "imbalance_ratio": None,
        "recommended_split": {},
        "warnings": [],
        "errors": [],
    }

    if not os.path.isdir(dataset_dir):
        msg = f"ERROR: Dataset directory does not exist: {dataset_dir}"
        print(msg)
        report["errors"].append(msg)
        return report

    # ── Per-class scan ────────────────────────────────────────────────────────
    all_filenames = []       # (class_name, filename) for duplicate detection
    all_md5s = defaultdict(list)  # md5 → list of (class, path)
    all_phashes = []         # (class, path, phash_str)
    all_widths, all_heights = [], []
    global_valid_count = 0
    global_corrupt_count = 0

    for cls_name in EXPECTED_CLASSES:
        cls_dir = os.path.join(dataset_dir, cls_name)
        cls_info = {
            "directory": cls_dir,
            "exists": os.path.isdir(cls_dir),
            "valid_images": 0,
            "corrupted_images": [],
            "widths": [],
            "heights": [],
            "formats": defaultdict(int),
        }

        if not cls_info["exists"]:
            report["classes_missing"].append(cls_name)
            report["per_class"][cls_name] = cls_info
            print(f"  [MISSING]  {cls_name}/  — directory not found")
            continue

        report["classes_found"].append(cls_name)
        files = sorted(os.listdir(cls_dir))
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]

        print(f"  [SCANNING] {cls_name}/  ({len(image_files)} files found)")

        for fname in image_files:
            fpath = os.path.join(cls_dir, fname)
            ext = os.path.splitext(fname)[1].lower()

            # Track filename for duplicate detection
            all_filenames.append((cls_name, fname))

            try:
                img = Image.open(fpath)
                img.verify()          # Check for corruption
                img = Image.open(fpath)  # Re-open after verify
                w, h = img.size
                fmt = img.format or ext.lstrip('.')

                cls_info["valid_images"] += 1
                cls_info["widths"].append(w)
                cls_info["heights"].append(h)
                cls_info["formats"][fmt.lower()] += 1
                report["format_distribution"][fmt.lower()] += 1
                all_widths.append(w)
                all_heights.append(h)
                global_valid_count += 1

                # MD5 for exact duplicate detection
                md5 = file_md5(fpath)
                if md5:
                    all_md5s[md5].append((cls_name, fpath))

                # Perceptual hash (optional, slower)
                if check_perceptual_duplicates:
                    ph = average_hash(img)
                    if ph:
                        all_phashes.append((cls_name, fpath, ph))

            except (UnidentifiedImageError, Exception) as e:
                cls_info["corrupted_images"].append({"file": fname, "error": str(e)})
                global_corrupt_count += 1

        # Summarise class widths/heights
        if cls_info["widths"]:
            cls_info["width_stats"] = {
                "min": int(min(cls_info["widths"])),
                "max": int(max(cls_info["widths"])),
                "mean": round(float(np.mean(cls_info["widths"])), 1),
            }
            cls_info["height_stats"] = {
                "min": int(min(cls_info["heights"])),
                "max": int(max(cls_info["heights"])),
                "mean": round(float(np.mean(cls_info["heights"])), 1),
            }
        else:
            cls_info["width_stats"] = {}
            cls_info["height_stats"] = {}

        del cls_info["widths"]
        del cls_info["heights"]
        cls_info["formats"] = dict(cls_info["formats"])

        report["per_class"][cls_name] = cls_info

    report["total_valid_images"] = global_valid_count
    report["total_corrupted"] = global_corrupt_count
    report["format_distribution"] = dict(report["format_distribution"])

    # ── Global dimension stats ─────────────────────────────────────────────────
    if all_widths:
        report["dimension_stats"] = {
            "width":  {"min": int(min(all_widths)),  "max": int(max(all_widths)),  "mean": round(float(np.mean(all_widths)), 1)},
            "height": {"min": int(min(all_heights)), "max": int(max(all_heights)), "mean": round(float(np.mean(all_heights)), 1)},
        }

    # ── Duplicate filename detection ───────────────────────────────────────────
    fname_only = [fname for _, fname in all_filenames]
    seen_fnames = defaultdict(list)
    for cls_name, fname in all_filenames:
        seen_fnames[fname].append(cls_name)
    dup_fnames = [(fname, classes) for fname, classes in seen_fnames.items() if len(classes) > 1]
    report["duplicate_filenames"] = dup_fnames

    # ── Exact duplicate content (MD5) ─────────────────────────────────────────
    exact_dups = {md5: paths for md5, paths in all_md5s.items() if len(paths) > 1}
    report["exact_duplicate_files"] = [
        {"md5": md5, "files": [(cls, p) for cls, p in paths]}
        for md5, paths in exact_dups.items()
    ]

    # ── Perceptual near-duplicate detection ───────────────────────────────────
    if check_perceptual_duplicates and len(all_phashes) > 1:
        print(f"\n  Computing perceptual hashes for {len(all_phashes)} images (may take time)...")
        near_dups = []
        for i in range(len(all_phashes)):
            for j in range(i + 1, len(all_phashes)):
                cls_i, path_i, ph_i = all_phashes[i]
                cls_j, path_j, ph_j = all_phashes[j]
                dist = hamming_distance(ph_i, ph_j)
                if dist <= phash_threshold:
                    near_dups.append({
                        "file_a": {"class": cls_i, "path": os.path.basename(path_i)},
                        "file_b": {"class": cls_j, "path": os.path.basename(path_j)},
                        "hamming_distance": dist
                    })
        report["perceptual_duplicate_pairs"] = near_dups

    # ── Class imbalance analysis ───────────────────────────────────────────────
    counts = {cls: report["per_class"][cls]["valid_images"]
              for cls in EXPECTED_CLASSES if cls in report["per_class"]}
    if counts:
        max_count = max(counts.values())
        nonzero = [v for v in counts.values() if v > 0]
        min_count = min(nonzero) if nonzero else 0
        if max_count == 0:
            report["imbalance_ratio"] = None  # No data yet
        elif min_count == 0:
            report["imbalance_ratio"] = None  # Some classes empty
        else:
            report["imbalance_ratio"] = round(max_count / min_count, 2)

    # ── Recommended split ─────────────────────────────────────────────────────
    n = global_valid_count
    if n > 0:
        train_n = int(round(n * 0.70))
        val_n   = int(round(n * 0.15))
        test_n  = n - train_n - val_n
        report["recommended_split"] = {
            "strategy": "Stratified (70% Train / 15% Val / 15% Test)",
            "train": train_n, "val": val_n, "test": test_n
        }

    # ── Warnings ──────────────────────────────────────────────────────────────
    if report["classes_missing"]:
        report["warnings"].append(f"Missing class directories: {report['classes_missing']}")
    if global_corrupt_count > 0:
        report["warnings"].append(f"{global_corrupt_count} corrupted/unreadable image(s) detected.")
    if dup_fnames:
        report["warnings"].append(f"{len(dup_fnames)} duplicate filename(s) found across classes (possible data leakage risk).")
    if report["exact_duplicate_files"]:
        report["warnings"].append(f"{len(report['exact_duplicate_files'])} group(s) of exactly identical files found.")

    return report


# ── Console report printer ────────────────────────────────────────────────────

def print_report(report: Dict):
    SEP  = '-' * 70
    SEP2 = '=' * 70
    print(f"\n{SEP2}")
    print(f"  SUMMARY")
    print(f"{SEP}")
    print(f"  Total valid images  : {report['total_valid_images']}")
    print(f"  Corrupted images    : {report['total_corrupted']}")
    print(f"  Classes found       : {report['classes_found']}")
    if report['classes_missing']:
        print(f"  [!] Classes MISSING : {report['classes_missing']}")

    print(f"\n{SEP}")
    print(f"  PER-CLASS BREAKDOWN")
    print(f"{SEP}")
    for cls_name in EXPECTED_CLASSES:
        info = report['per_class'].get(cls_name)
        if not info:
            continue
        count = info['valid_images']
        corrupt = len(info.get('corrupted_images', []))
        fmts = info.get('formats', {})
        w_stats = info.get('width_stats', {})
        h_stats = info.get('height_stats', {})
        print(f"\n  [{cls_name}]")
        print(f"    Valid images : {count}")
        if corrupt:
            print(f"    Corrupted    : {corrupt}  <-- REVIEW THESE FILES")
        if fmts:
            print(f"    Formats      : {fmts}")
        if w_stats:
            print(f"    Width        : min={w_stats['min']}  max={w_stats['max']}  mean={w_stats['mean']}")
            print(f"    Height       : min={h_stats['min']}  max={h_stats['max']}  mean={h_stats['mean']}")

    counts = {cls: report['per_class'].get(cls, {}).get('valid_images', 0) for cls in EXPECTED_CLASSES}
    if any(counts.values()):
        print(f"\n{SEP}")
        print(f"  CLASS DISTRIBUTION")
        print(f"{SEP}")
        total = sum(counts.values())
        max_c = max(counts.values())
        for cls, cnt in counts.items():
            bar_len = int((cnt / max(max_c, 1)) * 30)
            bar = '#' * bar_len
            pct = (cnt / max(total, 1)) * 100
            print(f"  {cls:<25} {cnt:>5}  {bar} {pct:.1f}%")
        print(f"  {'TOTAL':<25} {total:>5}")
        print(f"\n  Imbalance ratio (max/min) : {report['imbalance_ratio']}")
        if report['imbalance_ratio'] and report['imbalance_ratio'] > 3.0:
            print(f"  [!] WARNING: High class imbalance detected. Class-weighted loss is strongly recommended.")

    if report.get('format_distribution'):
        print(f"\n{SEP}")
        print(f"  FORMAT DISTRIBUTION (global)")
        print(f"{SEP}")
        for fmt, cnt in report['format_distribution'].items():
            print(f"  {fmt:<12}: {cnt}")

    if report.get('dimension_stats'):
        d = report['dimension_stats']
        print(f"\n{SEP}")
        print(f"  IMAGE DIMENSIONS (global)")
        print(f"{SEP}")
        print(f"  Width  - min: {d['width']['min']}, max: {d['width']['max']}, mean: {d['width']['mean']}")
        print(f"  Height - min: {d['height']['min']}, max: {d['height']['max']}, mean: {d['height']['mean']}")

    if report.get('duplicate_filenames'):
        print(f"\n{SEP}")
        print(f"  DUPLICATE FILENAMES ACROSS CLASSES  [{len(report['duplicate_filenames'])} found]")
        print(f"{SEP}")
        for fname, classes in report['duplicate_filenames'][:20]:
            print(f"  '{fname}'  appears in: {classes}")
        if len(report['duplicate_filenames']) > 20:
            print(f"  ... and {len(report['duplicate_filenames']) - 20} more.")

    if report.get('exact_duplicate_files'):
        print(f"\n{SEP}")
        print(f"  EXACT FILE DUPLICATES (identical content)  [{len(report['exact_duplicate_files'])} group(s)]")
        print(f"{SEP}")
        for dup in report['exact_duplicate_files'][:10]:
            paths_str = ', '.join(f"{cls}/{os.path.basename(p)}" for cls, p in dup['files'])
            print(f"  [MD5: {dup['md5'][:12]}...]  {paths_str}")
        if len(report['exact_duplicate_files']) > 10:
            print(f"  ... and {len(report['exact_duplicate_files']) - 10} more groups.")

    if report.get('perceptual_duplicate_pairs'):
        print(f"\n{SEP}")
        print(f"  NEAR-DUPLICATE IMAGES (perceptual hash)  [{len(report['perceptual_duplicate_pairs'])} pair(s)]")
        print(f"{SEP}")
        for pair in report['perceptual_duplicate_pairs'][:10]:
            print(f"  dist={pair['hamming_distance']}  {pair['file_a']['class']}/{pair['file_a']['path']}"
                  f"  <->  {pair['file_b']['class']}/{pair['file_b']['path']}")

    if report.get('recommended_split'):
        sp = report['recommended_split']
        print(f"\n{SEP}")
        print(f"  RECOMMENDED SPLIT  ({sp['strategy']})")
        print(f"{SEP}")
        print(f"  Train : {sp['train']}")
        print(f"  Val   : {sp['val']}")
        print(f"  Test  : {sp['test']}")

    if report.get('warnings'):
        print(f"\n{SEP}")
        print(f"  WARNINGS")
        print(f"{SEP}")
        for w in report['warnings']:
            print(f"  [!] {w}")

    print(f"\n{SEP2}")
    print(f"  Inspection complete.")
    print(f"{SEP2}\n")



# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspect Alzheimer MRI dataset directory before training."
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'data', 'dataset'),
        help='Path to dataset root directory containing class subdirectories.'
    )
    parser.add_argument(
        '--check_duplicates',
        action='store_true',
        help='Enable perceptual hash-based near-duplicate detection (slower for large datasets).'
    )
    parser.add_argument(
        '--save_report',
        type=str,
        default=None,
        help='Optional path to save the full inspection report as a JSON file.'
    )
    args = parser.parse_args()

    report = inspect_dataset(
        dataset_dir=args.dataset_dir,
        check_perceptual_duplicates=args.check_duplicates
    )

    print_report(report)

    if args.save_report:
        save_path = args.save_report
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        # Convert defaultdicts to plain dicts for JSON serialisation
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, default=str)
        print(f"  Report saved to: {save_path}\n")

    # Return non-zero exit code if dataset is not ready for training
    if report['total_valid_images'] == 0 or report['classes_missing']:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
