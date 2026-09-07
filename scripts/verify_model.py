"""
Phase 1 verification script — run this FIRST, before evaluate/benchmark/etc.

Confirms:
    - weights load correctly (strict=True in src.inference.load_model
      already enforces this — this script just makes it visible)
    - architecture matches checkpoint (implied by successful strict load)
    - model.eval() is set
    - a dummy forward pass produces the expected output shape
    - CPU works
    - CUDA works, if available
    - reports parameter count / trainable params / model size / input
      resolution, all read from the actual loaded model and checkpoint,
      never hand-typed

Run:
    python scripts/verify_model.py
"""

import sys
import time
import torch

sys.path.insert(0, ".")
from src.inference import load_model
from src.model import count_parameters, model_size_mb


def run_on(device_str):
    device = torch.device(device_str)
    print(f"\n--- Loading on {device_str} ---")

    t0 = time.time()
    model, classes, transform, device, checkpoint = load_model(device=device)
    load_time = time.time() - t0

    assert not model.training, "Model must be in eval mode after load_model()"

    image_size = checkpoint["preprocessing"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.inference_mode():
        out = model(dummy)

    expected_classes = checkpoint["architecture"]["num_classes"]
    assert out.shape == (1, expected_classes), (
        f"Output shape mismatch: got {tuple(out.shape)}, expected (1, {expected_classes})"
    )

    total_params, trainable_params = count_parameters(model)
    size_mb = model_size_mb(model)

    print(f"Load time:            {load_time:.2f}s")
    print(f"Classes ({len(classes)}):     {classes}")
    print(f"Input resolution:     {image_size}x{image_size}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (fp32):    {size_mb:.1f} MB")
    print(f"Output shape:         {tuple(out.shape)}  [OK]")
    print(f"eval() mode:          {not model.training}  [OK]")

    return {
        "device": device_str,
        "load_time_sec": load_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": size_mb,
        "num_classes": len(classes),
        "input_resolution": image_size,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL VERIFICATION")
    print("=" * 60)

    results = [run_on("cpu")]

    if torch.cuda.is_available():
        results.append(run_on("cuda"))
    else:
        print("\n--- CUDA not available on this machine, skipping GPU check ---")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
