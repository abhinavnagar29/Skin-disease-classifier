# Hybrid Skin Disease Classifier

A 22-class skin condition classifier combining ResNet50 and DenseNet121
feature extractors, with Grad-CAM interpretability, a real evaluation
pipeline, error analysis, calibration analysis, robustness testing, and
inference benchmarking — deployed as a Streamlit app.

> **Medical disclaimer:** This is a research/educational ML project.
> It is not a certified medical device, has not been clinically
> validated, and must never be used to make medical decisions. If you
> have a real skin concern, see a licensed dermatologist.

## Overview

Given a skin image, the model outputs a probability distribution over
22 categories and a Grad-CAM visualization showing which region of the
image the ResNet50 branch attended to. The project also includes the
evaluation, error-analysis, calibration, robustness, and benchmarking
infrastructure needed to actually characterize the model's behavior —
not just serve predictions.

## Key Results

**Not filled in yet — intentionally.** These numbers must come from
running the actual evaluation on the actual test set, not from an
estimate. Run:

```bash
python scripts/evaluate.py --test-dir data/test
```

then replace this block with the real contents of `results/metrics.json`.
Do not publish this README (or put numbers in a resume) with this
section still showing placeholders.

| Metric | Value |
|---|---|
| Accuracy | *(run evaluate.py)* |
| Balanced Accuracy | *(run evaluate.py)* |
| Macro F1 | *(run evaluate.py)* |
| Weighted F1 | *(run evaluate.py)* |
| Test set size | *(run evaluate.py)* |

## Architecture

```
Input (3 x H x W)
      │
      ├──────────→ ResNet50 (FC layer removed)      → GAP → 2048-d
      │
      └──────────→ DenseNet121 (classifier removed) → ReLU → GAP → 1024-d
                              │
                              ▼
                     concatenate → 3072-d
                              │
                              ▼
              Linear(3072→2048) → BatchNorm → ReLU → Dropout(0.5)
              Linear(2048→1024) → BatchNorm → ReLU → Dropout(0.5)
              Linear(1024→512)  → BatchNorm → ReLU → Dropout(0.5)
              Linear(512→num_classes)
                              │
                              ▼
                      logits → softmax
```

**Fusion is feature-level concatenation**, not a learned attention or
gating mechanism — stated precisely rather than oversold. Two
structurally different CNN families (residual connections vs. dense
connections) are used as complementary, independently pretrained
feature extractors, combined before the classification head.

Run `python scripts/verify_model.py` for the real parameter count,
model size, and load time — see `models/README.md` for why the
checkpoint lives on the Hugging Face Hub rather than in this repo.

## Dataset

**Not filled in yet.** Run:

```bash
python scripts/dataset_stats.py --train-dir data/train --val-dir data/val --test-dir data/test
```

and replace this section with the real contents of
`results/dataset_stats.json`, including:
- total images, per-split and per-class
- class imbalance ratio
- any cross-split leakage findings (the script actively checks for
  duplicate images appearing in more than one split — if it finds any,
  report that honestly here before reporting any metrics as clean)

## Methodology

Preprocessing (resize, normalization) is read directly from the
checkpoint's stored `preprocessing` config at inference time — see
`src/preprocessing.py` — so evaluation, benchmarking, and the deployed
app are guaranteed to use identical preprocessing, by construction,
not by convention.

Training methodology (loss function, optimizer, LR schedule, augmentation,
number of epochs) — **fill in from your actual training run/notebook.**
This project's scope, as delivered, covers evaluation and deployment of
an already-trained model; the training run itself happened separately
and its exact hyperparameters should be documented here from your
training logs.

## Evaluation

Run `python scripts/evaluate.py --test-dir data/test`, then paste the
real classification report and confusion matrix here. See
`results/README.md` for the full list of artifacts this produces.

## Error Analysis

Run `python scripts/error_analysis.py` (after `evaluate.py`), then
summarize `results/error_analysis/report.md` here: worst-performing
classes, most-confused class pairs, and characteristic high-confidence
failures.

## Interpretability

Grad-CAM is computed against `model.backbone.resnet_target_layer` (the
ResNet50 branch's final convolutional block) — see
`src/explainability.py`. Scope, stated explicitly:

- Explains the ResNet50 branch's attention, not the DenseNet branch or
  the fused decision as a whole.
- Shows correlation between image regions and the predicted class's
  gradient signal — it does **not** demonstrate that the model is using
  clinically meaningful features (asymmetry, texture, border
  irregularity, etc.). Don't claim more than this.

## Robustness

Run `python scripts/robustness.py --test-dir data/test`, then paste the
real flip-rate table from `results/robustness/robustness_summary.csv`
here — brightness/contrast jitter, small rotation, downscale-upscale,
and JPEG compression at quality 30, tested against real test images.

## Calibration

Run `python scripts/calibration.py --test-dir data/test --val-dir data/val`,
then report the real ECE from `results/calibration/ece_report.json`.
Temperature scaling's `T` is fit on the validation set, never on the
test set, to avoid leaking test information into a "calibrated" number
— see the docstring in `scripts/calibration.py` for why that distinction
matters.

## Inference Benchmark

Run `python scripts/benchmark.py`, then paste real latency/throughput
numbers from `results/benchmark.json` here. CPU numbers will always be
available; GPU numbers only if you have CUDA locally — Streamlit
Community Cloud itself is CPU-only, so the CPU numbers are what the
deployed app will actually experience.

## Demo

Streamlit deployment: *(fill in your `https://<app-name>.streamlit.app` URL after deploying — see Deployment section)*

## Installation

```bash
git clone https://github.com/abhinavnagar29/Skin-disease-classifier.git
cd Skin-disease-classifier

python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

First run downloads the checkpoint from the Hugging Face Hub — needs
internet access once. `torch.cuda.is_available()` handles the CPU/GPU
switch automatically; no configuration needed either way.

## Evaluation

```bash
python scripts/verify_model.py
python scripts/evaluate.py --test-dir data/test
python scripts/error_analysis.py
python scripts/robustness.py --test-dir data/test
python scripts/calibration.py --test-dir data/test --val-dir data/val
python scripts/benchmark.py
```

## Project Structure

```
Skin-disease-classifier/
├── app.py                  - Streamlit application
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── model.py             - architecture (unchanged from trained checkpoint)
│   ├── preprocessing.py     - transform built from checkpoint config
│   ├── inference.py         - load_model() + predict(), single source of truth
│   └── explainability.py    - Grad-CAM
│
├── scripts/
│   ├── verify_model.py      - Phase 1 sanity checks
│   ├── dataset_stats.py     - Phase 3 dataset analysis + leakage check
│   ├── evaluate.py          - Phase 2 full test-set evaluation
│   ├── error_analysis.py    - Phase 4, built on evaluate.py's output
│   ├── robustness.py        - Phase 6 perturbation testing
│   ├── calibration.py       - Phase 7 ECE + temperature scaling
│   └── benchmark.py         - Phase 8 latency/throughput
│
├── results/                 - generated by scripts above, not committed with fake data
├── assets/                  - images/diagrams for this README
└── models/
    └── README.md            - why weights live on the HF Hub, not in git
```

## Limitations

- Grad-CAM covers the ResNet50 branch only — the DenseNet branch and
  the fused decision are not directly visualized.
- Robustness testing covers common image-quality perturbations
  (brightness, contrast, rotation, resize, compression) — it is not
  adversarial robustness testing.
- No fairness/subgroup analysis (e.g., across skin tones) has been
  performed — treat any deployment beyond research/demo use with real
  caution.
- *(Add real limitations found from actually running the evaluation —
  e.g., specific weak classes, dataset imbalance, calibration issues —
  once the scripts have been run.)*

## Medical Disclaimer

This project is a research and educational demonstration only. It is
not a certified medical device, has not undergone clinical validation,
and predictions should never be used to make medical decisions. Always
consult a licensed dermatologist or physician for any real skin concern.
