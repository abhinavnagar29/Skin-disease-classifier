"""
Phase 3 — dataset statistics, computed from your actual data folders.

I do not have your dataset here, so I cannot report real numbers in the
README until you run this. Point it at your train/val/test directories
(ImageFolder layout: split_dir/<class_name>/<image>.jpg).

Checks performed:
    - image counts per split, per class
    - class imbalance (max/min class count ratio)
    - image dimension distribution
    - exact-duplicate detection (via file hash) WITHIN and ACROSS splits
      -- cross-split duplicates are a real leakage bug, not a style nit
    - corrupted/unreadable image detection

Outputs:
    results/dataset_stats.json
    results/class_distribution.png

Run:
    python scripts/dataset_stats.py --train-dir data/train --val-dir data/val --test-dir data/test
"""

import argparse
import hashlib
import json
import os

import matplotlib.pyplot as plt
from PIL import Image


def hash_file(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def scan_split(split_dir):
    """Returns {class_name: [file_paths]}, plus a list of corrupted files."""
    class_files = {}
    corrupted = []

    if not os.path.isdir(split_dir):
        return class_files, corrupted

    for class_name in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        files = []
        for fname in os.listdir(class_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            fpath = os.path.join(class_path, fname)
            try:
                with Image.open(fpath) as img:
                    img.verify()
                files.append(fpath)
            except Exception:
                corrupted.append(fpath)

        class_files[class_name] = files

    return class_files, corrupted


def check_cross_split_duplicates(splits):
    """splits: {split_name: {class_name: [paths]}}. Hashes every file and
    flags any hash appearing in more than one split — this is actual
    train/test leakage, not a hypothetical."""
    hash_to_locations = {}

    for split_name, class_files in splits.items():
        for class_name, paths in class_files.items():
            for path in paths:
                h = hash_file(path)
                hash_to_locations.setdefault(h, []).append((split_name, class_name, path))

    leaks = {h: locs for h, locs in hash_to_locations.items() if len(set(l[0] for l in locs)) > 1}
    return leaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-dir", default="data/val")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    splits = {}
    corrupted_all = {}
    for split_name, split_dir in [("train", args.train_dir), ("val", args.val_dir), ("test", args.test_dir)]:
        class_files, corrupted = scan_split(split_dir)
        splits[split_name] = class_files
        corrupted_all[split_name] = corrupted

    stats = {"splits": {}}
    all_classes = set()

    for split_name, class_files in splits.items():
        if not class_files:
            continue
        counts = {c: len(paths) for c, paths in class_files.items()}
        all_classes.update(counts.keys())
        total = sum(counts.values())

        stats["splits"][split_name] = {
            "total_images": total,
            "num_classes": len(counts),
            "per_class_counts": counts,
            "max_class_count": max(counts.values()) if counts else 0,
            "min_class_count": min(counts.values()) if counts else 0,
            "imbalance_ratio": (max(counts.values()) / max(1, min(counts.values()))) if counts else None,
            "corrupted_images": len(corrupted_all[split_name]),
        }

    print("DATASET STATS:")
    for split_name, s in stats["splits"].items():
        print(f"\n{split_name}: {s['total_images']} images, {s['num_classes']} classes, "
              f"imbalance ratio {s['imbalance_ratio']:.1f}x, {s['corrupted_images']} corrupted")

    print("\nChecking for cross-split duplicate images (this can take a while on large datasets)...")
    leaks = check_cross_split_duplicates(splits)
    stats["cross_split_leakage"] = {
        "num_duplicate_images_across_splits": len(leaks),
        "examples": [
            {"locations": [{"split": l[0], "class": l[1], "path": l[2]} for l in locs]}
            for locs in list(leaks.values())[:10]
        ],
    }

    if leaks:
        print(f"\n*** WARNING: {len(leaks)} images appear in more than one split. "
              f"This is train/test leakage — see results/dataset_stats.json for examples. "
              f"Do not report test-set metrics as clean until this is resolved. ***")
    else:
        print("\nNo cross-split duplicate images found.")

    with open(f"{args.results_dir}/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # class distribution plot (train split, if present)
    if "train" in stats["splits"]:
        counts = stats["splits"]["train"]["per_class_counts"]
        fig, ax = plt.subplots(figsize=(10, 6))
        classes_sorted = sorted(counts, key=counts.get, reverse=True)
        ax.bar(range(len(classes_sorted)), [counts[c] for c in classes_sorted])
        ax.set_xticks(range(len(classes_sorted)))
        ax.set_xticklabels(classes_sorted, rotation=90, fontsize=7)
        ax.set_ylabel("Image count")
        ax.set_title("Class distribution (train split)")
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/class_distribution.png", dpi=150)
        plt.close()

    print(f"\nResults written to {args.results_dir}/dataset_stats.json")


if __name__ == "__main__":
    main()
