"""
Phase 2 — real evaluation on the held-out test set.

Supports two dataset layouts via --format:
    imagefolder (default): --test-dir points to a folder structured as
        test_dir/<class_name>/<image>.jpg   (standard torchvision.datasets.ImageFolder layout)
    csv: --test-csv points to a CSV with columns: image_path,label

If your dataset isn't in either layout, adjust `load_test_set()` below —
everything downstream (metrics, plots, saving) is layout-agnostic.

Class order is read from the checkpoint (checkpoint["class_names"]), and
ImageFolder's alphabetical class order is remapped to match it — this
matters: if your folder names don't exactly match checkpoint class names,
predictions will silently misalign. The script asserts this explicitly
rather than failing silently.

Outputs (all under results/):
    metrics.json             - every metric requested in Phase 2
    classification_report.csv
    confusion_matrix.png
    per_class_f1.png
    predictions.csv          - per-image: true label, predicted label,
                                confidence, top-5 predictions
    confidence_distribution.png

Run:
    python scripts/evaluate.py --test-dir data/test
    python scripts/evaluate.py --test-csv data/test_labels.csv
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, classification_report,
    top_k_accuracy_score,
)
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.inference import load_model, predict
from src.preprocessing import preprocess_image


def load_test_set_imagefolder(test_dir, classes):
    from torchvision.datasets import ImageFolder
    ds = ImageFolder(test_dir)

    # ImageFolder assigns class indices alphabetically by folder name.
    # Remap to the checkpoint's class order so predicted indices line up.
    folder_classes = ds.classes
    if set(folder_classes) != set(classes):
        missing = set(classes) - set(folder_classes)
        extra = set(folder_classes) - set(classes)
        raise ValueError(
            f"Test folder class names don't match checkpoint class_names.\n"
            f"In checkpoint but not in folder: {missing}\n"
            f"In folder but not in checkpoint: {extra}\n"
            f"Fix folder names or checkpoint class list before evaluating — "
            f"do NOT proceed with a silent misalignment."
        )

    folder_idx_to_name = {v: k for k, v in ds.class_to_idx.items()}
    samples = [(path, folder_idx_to_name[idx]) for path, idx in ds.samples]
    return samples


def load_test_set_csv(csv_path):
    df = pd.read_csv(csv_path)
    assert "image_path" in df.columns and "label" in df.columns, (
        "CSV must have columns: image_path,label"
    )
    return list(zip(df["image_path"], df["label"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["imagefolder", "csv"], default="imagefolder")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import os
    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading model...")
    model, classes, transform, device, checkpoint = load_model()
    print(f"Device: {device}, classes: {len(classes)}")

    if args.format == "imagefolder":
        samples = load_test_set_imagefolder(args.test_dir, classes)
    else:
        samples = load_test_set_csv(args.test_csv)

    print(f"Evaluating {len(samples)} test images...")

    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_true, y_pred, y_conf = [], [], []
    all_probs = []
    rows = []

    t0 = time.time()
    with torch.inference_mode():
        for i, (path, true_label) in enumerate(samples):
            image = Image.open(path).convert("RGB")
            tensor = preprocess_image(image, transform).to(device)

            top5, probs = predict(model, tensor, classes, top_k=5)

            pred_label = max(top5, key=top5.get)
            confidence = top5[pred_label]

            y_true.append(class_to_idx[true_label])
            y_pred.append(class_to_idx[pred_label])
            y_conf.append(confidence)
            all_probs.append(probs.cpu().numpy())

            rows.append({
                "image_path": path,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "correct": true_label == pred_label,
                "top5": ",".join(top5.keys()),
            })

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(samples)}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/len(samples)*1000:.1f} ms/image)")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_probs = np.array(all_probs)

    # ---- metrics ----
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    present_labels = sorted(set(y_true.tolist()))
    top3_acc = None
    top5_acc = None
    try:
        top3_acc = top_k_accuracy_score(y_true, all_probs, k=3, labels=list(range(len(classes))))
        top5_acc = top_k_accuracy_score(y_true, all_probs, k=5, labels=list(range(len(classes))))
    except Exception as e:
        print(f"Top-k accuracy skipped: {e}")

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "num_test_images": len(samples),
        "num_classes": len(classes),
        "mean_confidence": float(np.mean(y_conf)),
        "eval_time_sec": elapsed,
    }

    with open(f"{args.results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nMETRICS:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ---- classification report ----
    report = classification_report(
        y_true, y_pred, labels=list(range(len(classes))), target_names=classes,
        output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(f"{args.results_dir}/classification_report.csv")

    # ---- confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (n={len(samples)})")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    np.save(f"{args.results_dir}/confusion_matrix.npy", cm)

    # ---- per-class F1 ----
    per_class_f1 = report_to_series(report, classes, "f1-score")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(classes)), per_class_f1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1")
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/per_class_f1.png", dpi=150)
    plt.close()

    # ---- confidence distribution ----
    fig, ax = plt.subplots(figsize=(8, 5))
    correct_conf = [c for c, r in zip(y_conf, rows) if r["correct"]]
    wrong_conf = [c for c, r in zip(y_conf, rows) if not r["correct"]]
    ax.hist(correct_conf, bins=20, alpha=0.6, label="Correct predictions")
    ax.hist(wrong_conf, bins=20, alpha=0.6, label="Incorrect predictions")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Confidence distribution: correct vs incorrect")
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/confidence_distribution.png", dpi=150)
    plt.close()

    # ---- predictions.csv (feeds error_analysis.py) ----
    pd.DataFrame(rows).to_csv(f"{args.results_dir}/predictions.csv", index=False)

    print(f"\nAll results written to {args.results_dir}/")


def report_to_series(report_dict, classes, key):
    return [report_dict[c][key] for c in classes]


if __name__ == "__main__":
    main()
