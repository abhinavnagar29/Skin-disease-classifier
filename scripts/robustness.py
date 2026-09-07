"""
Phase 6 — robustness to controlled perturbations.

Takes a sample of real test images and, for each, compares the
prediction on the original image against predictions on perturbed
versions. Measures:
    - prediction flip rate (%) per perturbation type
    - mean confidence delta per perturbation type

This is NOT adversarial robustness (no gradient-based attacks) — it's
robustness to the kind of image-quality variation a real user's upload
might have (phone camera brightness, slight rotation, compression from
messaging apps, etc.). State that scope explicitly, don't imply more.

Run:
    python scripts/robustness.py --test-dir data/test --n-samples 100
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
import io

sys.path.insert(0, ".")
from src.inference import load_model, predict
from src.preprocessing import preprocess_image


def perturb_brightness(img, factor=1.4):
    return ImageEnhance.Brightness(img).enhance(factor)


def perturb_contrast(img, factor=1.4):
    return ImageEnhance.Contrast(img).enhance(factor)


def perturb_rotation(img, degrees=10):
    return img.rotate(degrees, expand=False, fillcolor=(128, 128, 128))


def perturb_downscale_upscale(img, factor=0.5):
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def perturb_jpeg_compression(img, quality=30):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


PERTURBATIONS = {
    "brightness_+40%": lambda img: perturb_brightness(img, 1.4),
    "contrast_+40%": lambda img: perturb_contrast(img, 1.4),
    "rotation_10deg": lambda img: perturb_rotation(img, 10),
    "downscale_upscale_0.5x": lambda img: perturb_downscale_upscale(img, 0.5),
    "jpeg_quality_30": lambda img: perturb_jpeg_compression(img, 30),
}


def collect_sample_images(test_dir, n_samples, seed=42):
    paths = []
    for root, _, files in os.walk(test_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                paths.append(os.path.join(root, fname))
    random.Random(seed).shuffle(paths)
    return paths[:n_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--results-dir", default="results/robustness")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading model...")
    model, classes, transform, device, checkpoint = load_model()

    image_paths = collect_sample_images(args.test_dir, args.n_samples)
    print(f"Testing robustness on {len(image_paths)} sampled images x {len(PERTURBATIONS)} perturbations")

    rows = []
    with torch.inference_mode():
        for path in image_paths:
            original = Image.open(path).convert("RGB")
            tensor = preprocess_image(original, transform).to(device)
            orig_top1_dict, orig_probs = predict(model, tensor, classes, top_k=1)
            orig_pred = next(iter(orig_top1_dict))
            orig_conf = orig_top1_dict[orig_pred]

            for pert_name, pert_fn in PERTURBATIONS.items():
                perturbed = pert_fn(original)
                p_tensor = preprocess_image(perturbed, transform).to(device)
                pert_top1_dict, pert_probs = predict(model, p_tensor, classes, top_k=1)
                pert_pred = next(iter(pert_top1_dict))
                pert_conf = pert_top1_dict[pert_pred]

                rows.append({
                    "image_path": path,
                    "perturbation": pert_name,
                    "original_pred": orig_pred,
                    "original_conf": orig_conf,
                    "perturbed_pred": pert_pred,
                    "perturbed_conf": pert_conf,
                    "prediction_flipped": orig_pred != pert_pred,
                    "confidence_delta": pert_conf - orig_conf,
                })

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.results_dir}/robustness_results.csv", index=False)

    summary = df.groupby("perturbation").agg(
        flip_rate_pct=("prediction_flipped", lambda x: 100 * x.mean()),
        mean_confidence_delta=("confidence_delta", "mean"),
        std_confidence_delta=("confidence_delta", "std"),
    ).reset_index()
    summary.to_csv(f"{args.results_dir}/robustness_summary.csv", index=False)

    print("\nROBUSTNESS SUMMARY:")
    print(summary.to_string(index=False))
    print(f"\nFull results written to {args.results_dir}/")


if __name__ == "__main__":
    main()
