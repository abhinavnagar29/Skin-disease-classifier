"""
Phase 7 — calibration: is raw softmax confidence meaningful?

Correct methodology, stated explicitly:
    - ECE and the reliability diagram are computed on the TEST set logits.
    - Temperature scaling's temperature T is fit on a separate VALIDATION
      set (--val-dir), never on the test set — fitting T on test data
      would leak test-set information into a "post-hoc calibrated"
      number, invalidating it. If you don't have a separate val split,
      this script will say so and skip temperature scaling rather than
      quietly fitting T on the test set.

Outputs under results/calibration/:
    ece_report.json           - raw ECE, and calibrated ECE if val set given
    reliability_diagram.png   - raw (and calibrated, if available)
    temperature.json          - the fitted scalar T, if computed

Run:
    python scripts/calibration.py --test-dir data/test --val-dir data/val
    python scripts/calibration.py --test-dir data/test     # ECE only, no temp scaling
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.inference import load_model
from src.preprocessing import preprocess_image


def collect_logits_and_labels(model, image_label_pairs, transform, classes, device):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for path, true_label in image_label_pairs:
            image = Image.open(path).convert("RGB")
            tensor = preprocess_image(image, transform).to(device)
            logits = model(tensor)
            all_logits.append(logits.cpu())
            all_labels.append(class_to_idx[true_label])
    return torch.cat(all_logits, dim=0), torch.tensor(all_labels)


def expected_calibration_error(probs, labels, n_bins=15):
    """Standard binned ECE: weighted average |confidence - accuracy| per bin."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    bin_stats = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
            bin_stats.append({
                "bin_range": f"{lo:.2f}-{hi:.2f}",
                "accuracy": acc_in_bin.item(),
                "confidence": conf_in_bin.item(),
                "count": in_bin.sum().item(),
            })

    return ece.item(), bin_stats


def fit_temperature(val_logits, val_labels):
    """Fits a single scalar T minimizing NLL on the validation set (Guo et al., 2017)."""
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(val_logits / temperature, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def reliability_diagram(bin_stats_raw, bin_stats_cal, out_path):
    fig, axes = plt.subplots(1, 2 if bin_stats_cal else 1, figsize=(12 if bin_stats_cal else 6, 5))
    if not bin_stats_cal:
        axes = [axes]

    for ax, stats, title in zip(
        axes,
        [bin_stats_raw, bin_stats_cal] if bin_stats_cal else [bin_stats_raw],
        ["Raw confidence", "Temperature-scaled"] if bin_stats_cal else ["Raw confidence"],
    ):
        confs = [s["confidence"] for s in stats]
        accs = [s["accuracy"] for s in stats]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.scatter(confs, accs, label="Model")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--val-dir", default=None, help="Separate val set for fitting temperature. Omit to skip temperature scaling.")
    parser.add_argument("--results-dir", default="results/calibration")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    from torchvision.datasets import ImageFolder

    model, classes, transform, device, checkpoint = load_model()

    test_ds = ImageFolder(args.test_dir)
    idx_to_name = {v: k for k, v in test_ds.class_to_idx.items()}
    test_pairs = [(path, idx_to_name[idx]) for path, idx in test_ds.samples]

    print(f"Collecting logits for {len(test_pairs)} test images...")
    test_logits, test_labels = collect_logits_and_labels(model, test_pairs, transform, classes, device)
    test_probs = F.softmax(test_logits, dim=1)

    raw_ece, raw_bins = expected_calibration_error(test_probs, test_labels)
    print(f"Raw ECE: {raw_ece:.4f}")

    result = {"raw_ece": raw_ece, "num_test_images": len(test_pairs), "n_bins": 15}
    calibrated_bins = None

    if args.val_dir:
        val_ds = ImageFolder(args.val_dir)
        val_idx_to_name = {v: k for k, v in val_ds.class_to_idx.items()}
        val_pairs = [(path, val_idx_to_name[idx]) for path, idx in val_ds.samples]

        print(f"Fitting temperature on {len(val_pairs)} validation images...")
        val_logits, val_labels = collect_logits_and_labels(model, val_pairs, transform, classes, device)

        T = fit_temperature(val_logits, val_labels)
        print(f"Fitted temperature: {T:.3f}")

        calibrated_probs = F.softmax(test_logits / T, dim=1)
        cal_ece, calibrated_bins = expected_calibration_error(calibrated_probs, test_labels)
        print(f"Calibrated ECE (T={T:.3f}): {cal_ece:.4f}")

        result["temperature"] = T
        result["calibrated_ece"] = cal_ece
        result["ece_improvement"] = raw_ece - cal_ece

        with open(f"{args.results_dir}/temperature.json", "w") as f:
            json.dump({"temperature": T}, f, indent=2)
    else:
        print("No --val-dir given: reporting raw ECE only, skipping temperature scaling "
              "(fitting T on the test set would invalidate the calibrated number).")

    with open(f"{args.results_dir}/ece_report.json", "w") as f:
        json.dump(result, f, indent=2)

    reliability_diagram(raw_bins, calibrated_bins, f"{args.results_dir}/reliability_diagram.png")

    print(f"\nResults written to {args.results_dir}/")


if __name__ == "__main__":
    main()
