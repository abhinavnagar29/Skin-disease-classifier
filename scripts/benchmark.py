"""
Phase 8 — inference benchmarking, with correct CUDA timing methodology.

Correct methodology used here (stated explicitly since it's easy to get
wrong and worth being able to explain in an interview):
    - Warmup iterations before timing (first CUDA calls include kernel
      compilation/caching overhead — timing them would overstate latency).
    - torch.cuda.synchronize() before starting AND before stopping the
      timer for GPU runs — CUDA calls are asynchronous, so without
      synchronize() you'd measure "time to queue the op," not "time to
      actually finish it."
    - torch.inference_mode() + model.eval() for every timed run.
    - Batch throughput measured separately from single-image latency —
      they are different numbers and conflating them is a common mistake.

Outputs:
    results/benchmark.json

Run:
    python scripts/benchmark.py
"""

import json
import sys
import time

import torch

sys.path.insert(0, ".")
from src.inference import load_model


N_WARMUP = 10
N_TIMED = 50
BATCH_SIZES = [1, 4, 8, 16]


def time_single_image_latency(model, device, image_size, n_warmup=N_WARMUP, n_timed=N_TIMED):
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_timed):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "p50_ms": sorted(times_ms)[len(times_ms) // 2],
    }


def time_batch_throughput(model, device, image_size, batch_size, n_warmup=5, n_timed=20):
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_timed):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    total_images = batch_size * n_timed
    return {
        "batch_size": batch_size,
        "images_per_sec": total_images / elapsed,
        "sec_per_batch": elapsed / n_timed,
    }


def benchmark_on_device(device_str, image_size):
    device = torch.device(device_str)
    print(f"\n--- Benchmarking on {device_str} ---")

    t0 = time.time()
    model, classes, transform, device, checkpoint = load_model(device=device)
    load_time = time.time() - t0
    print(f"Model load time: {load_time:.2f}s")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    latency = time_single_image_latency(model, device, image_size)
    print(f"Single-image latency: {latency['mean_ms']:.2f}ms mean (p50={latency['p50_ms']:.2f}ms)")

    throughput = []
    for bs in BATCH_SIZES:
        r = time_batch_throughput(model, device, image_size, bs)
        throughput.append(r)
        print(f"  batch={bs}: {r['images_per_sec']:.1f} images/sec")

    result = {
        "device": device_str,
        "model_load_time_sec": load_time,
        "single_image_latency": latency,
        "batch_throughput": throughput,
    }

    if device.type == "cuda":
        result["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory: {result['peak_gpu_memory_mb']:.1f} MB")

    return result


def main():
    from src.inference import load_model as _lm
    _, _, _, _, checkpoint = _lm(device=torch.device("cpu"))
    image_size = checkpoint["preprocessing"]["image_size"]

    results = {"input_resolution": image_size, "runs": []}

    results["runs"].append(benchmark_on_device("cpu", image_size))

    if torch.cuda.is_available():
        results["runs"].append(benchmark_on_device("cuda", image_size))
    else:
        print("\nCUDA not available on this machine — GPU numbers not collected. "
              "Note this explicitly in the README rather than omitting the section.")

    with open("results/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark written to results/benchmark.json")


if __name__ == "__main__":
    main()
