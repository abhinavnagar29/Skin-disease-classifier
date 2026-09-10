# Hybrid Skin Disease Classifier

A 22-class skin condition classifier combining ResNet50 and DenseNet121
feature extractors, with Grad-CAM interpretability and a full evaluation,
error-analysis, and calibration pipeline — deployed as a Streamlit app.

**Live demo:** https://skin-disease-classifier-fq9tu2ysuqwme9l9gds9xc.streamlit.app/

> **Medical disclaimer:** This is a research/educational ML project. It is
> not a certified medical device, has not been clinically validated, and
> must never be used to make medical decisions. If you have a real skin
> concern, see a licensed dermatologist.

---

## Overview

Given a skin image, the model outputs a probability distribution over 22
dermatological categories and a Grad-CAM visualization showing which
region of the image the ResNet50 branch attended to.

## Key Results

Evaluated on a held-out test set of **1,546 images across 22 classes**.

| Metric                    | Value  |
| -------------------------- | ------ |
| Accuracy                    | 75.55% |
| Balanced Accuracy            | 72.52% |
| Macro Precision               | 73.33% |
| Macro Recall                   | 72.52% |
| Macro F1                        | 72.37% |
| Weighted Precision               | 76.51% |
| Weighted Recall                   | 75.55% |
| Weighted F1                        | 75.54% |
| Top-3 Accuracy                      | 90.36% |
| Top-5 Accuracy                       | 94.76% |
| Mean predicted confidence             | 88.70% |

Top-3/Top-5 accuracy (90.4% / 94.8%) substantially exceeds top-1 accuracy,
showing that the correct class frequently remains among the model's highest-ranked predictions.

The checkpoint was selected using **validation macro recall = 73.21%**
on a **15% validation split (seed 42)**. The held-out test set was used
only for final evaluation.

## Comprehensive Evaluation Methodology

This project goes beyond simple accuracy reporting with a complete evaluation pipeline:

- **Model Verification**: Parameter count, model size, load time, and forward pass validation
- **Dataset Statistics**: Class distribution, imbalance analysis, corrupted image detection
- **Full Test Evaluation**: Accuracy, precision, recall, F1, top-k accuracy, confusion matrix
- **Error Analysis**: Per-class breakdown, most confused pairs, high/low confidence predictions
- **Calibration Analysis**: Expected Calibration Error (ECE), reliability diagrams, temperature scaling
- **Robustness Testing**: Sensitivity to brightness, contrast, rotation, scaling, JPEG compression
- **Performance Benchmarking**: Latency, throughput, batch size optimization

All evaluation scripts are designed with correct methodology:
- Test set never used during training or model selection
- Calibration temperature fitted on validation set only (never test set)
- Proper CUDA synchronization for accurate timing
- Warmup iterations before benchmarking

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

Fusion is feature-level concatenation, not a learned attention or gating
mechanism. Two structurally different CNN families (residual connections
vs. dense connections) act as complementary, independently pretrained
feature extractors, combined before the classification head.

Run `python scripts/verify_model.py` for the real parameter count, model
size, and load time — see `models/README.md` for why the checkpoint lives
on the Hugging Face Hub rather than in this repo.

## Methodology

Preprocessing (resize, normalization) is read directly from the
checkpoint's stored `preprocessing` config at inference time — see
`src/preprocessing.py` — so evaluation and deployment use the same
checkpoint-defined preprocessing configuration.

## Evaluation

Run: `python scripts/evaluate.py --test-dir data/test`

### Per-class results

| Class                | Precision | Recall | F1-score | Support |
| --------------------- | --------- | ------ | -------- | ------- |
| Acne                   | 0.829     | 0.892  | 0.859    | 65      |
| Actinic_Keratosis      | 0.747     | 0.783  | 0.765    | 83      |
| Benign_tumors          | 0.633     | 0.785  | 0.701    | 121     |
| Bullous                | 0.844     | 0.691  | 0.760    | 55      |
| Candidiasis            | 0.679     | 0.704  | 0.691    | 27      |
| DrugEruption           | 0.662     | 0.705  | 0.683    | 61      |
| Eczema                 | 0.781     | 0.732  | 0.756    | 112     |
| Infestations_Bites     | 0.784     | 0.483  | 0.598    | 60      |
| Lichen                 | 0.683     | 0.672  | 0.678    | 61      |
| Lupus                  | 0.643     | 0.529  | 0.581    | 34      |
| Moles                  | 0.585     | 0.600  | 0.593    | 40      |
| Psoriasis              | 0.792     | 0.693  | 0.739    | 88      |
| Rosacea                | 0.667     | 0.786  | 0.721    | 28      |
| Seborrh_Keratoses      | 0.850     | 0.667  | 0.747    | 51      |
| SkinCancer             | 0.750     | 0.584  | 0.657    | 77      |
| Sun_Sunlight_Damage    | 0.595     | 0.647  | 0.620    | 34      |
| Tinea                  | 0.768     | 0.745  | 0.756    | 102     |
| Unknown_Normal         | 0.984     | 0.958  | 0.971    | 189     |
| Vascular_Tumors        | 0.577     | 0.750  | 0.652    | 60      |
| Vasculitis             | 0.575     | 0.808  | 0.672    | 52      |
| Vitiligo               | 0.894     | 0.927  | 0.910    | 82      |
| Warts                  | 0.813     | 0.813  | 0.813    | 64      |
| **Accuracy**           |           |        | **0.755**| 1546    |
| **Macro avg**          | 0.733     | 0.725  | 0.724    | 1546    |
| **Weighted avg**       | 0.765     | 0.755  | 0.755    | 1546    |

Full artifacts: `results/metrics.json`, `results/classification_report.csv`,
`results/confusion_matrix.png`, `results/per_class_f1.png`,
`results/confidence_distribution.png`, `results/predictions.csv`
(per-image true label, predicted label, confidence, top-5 predictions).

## Error Analysis

Run: `python scripts/error_analysis.py`

- **1,168 / 1,546 (75.5%) correct, 378 (24.5%) wrong.**
- **Weakest classes by F1:** `Lupus` (0.581), `Moles` (0.593),
  `Infestations_Bites` (0.598), `Sun_Sunlight_Damage` (0.620),
  `Vascular_Tumors` (0.652).
- **Most confused class pairs** (true → predicted, count): `SkinCancer →
  Benign_tumors` (13), `Vascular_Tumors → Benign_tumors` (9), `Eczema →
  Tinea` (8), `Eczema → Psoriasis` (7), `Moles → Benign_tumors` (7),
  `Unknown_Normal → Vitiligo` (7). `Benign_tumors` is a recurring
  fallback prediction for several visually similar lesion classes — the
  `SkinCancer → Benign_tumors` direction is the most clinically
  significant one to address further.
- **High-confidence wrong predictions:** mean confidence on wrong
  predictions is 0.722 — the model is sometimes confidently incorrect,
  not just uncertain-and-wrong (top 20 in
  `results/error_analysis/high_confidence_errors.csv`).
- **Low-confidence correct predictions:** mean confidence on correct
  predictions is 0.941 (bottom 20 in
  `results/error_analysis/low_confidence_correct.csv`).

Full report: `results/error_analysis/report.md`, plus
`worst_classes.csv`, `most_confused_pairs.csv`,
`high_confidence_errors.csv`, `low_confidence_correct.csv`.

## Interpretability

Grad-CAM is computed against `model.backbone.resnet_target_layer` (the
ResNet50 branch's final convolutional block) — see
`src/explainability.py`. It explains the ResNet50 branch's attention,
not the DenseNet branch or the fused decision as a whole, and shows
correlation between image regions and the predicted class's gradient
signal — not that the model is using clinically meaningful features
(asymmetry, texture, border irregularity, etc.).

## Calibration

Run: `python scripts/calibration.py --test-dir data/test --val-dir data/val`

| Metric               | Value  |
| ---------------------- | ------ |
| Raw ECE (15 bins)       | 0.1322 |

Full artifacts: `results/calibration/ece_report.json`,
`results/calibration/reliability_diagram.png`.

Mean predicted confidence over the test set is 88.7% while actual
accuracy is 75.5% — the raw ECE of 0.132 quantifies that gap: the
model's softmax outputs are overconfident and shouldn't be read as
calibrated probabilities out of the box. Temperature scaling (fit on a
separate validation split, never on the test set, to avoid leaking test
information into the calibrated number) is supported by
`scripts/calibration.py` via `--val-dir`.

## Robustness Testing

Run: `python scripts/robustness.py --test-dir data/test --n-samples 100`

Tested on 50 sampled images across 5 perturbation types:

| Perturbation               | Flip Rate | Mean Confidence Delta |
| -------------------------- | --------- | --------------------- |
| Brightness +40%            | 12.0%     | -0.047                |
| Contrast +40%              | 10.0%     | -0.017                |
| Rotation 10°               | 16.0%     | -0.019                |
| Downscale/Upscale 0.5x      | 12.0%     | -0.015                |
| JPEG Quality 30            | 38.0%     | -0.056                |

The model is most sensitive to JPEG compression artifacts (38% flip rate at quality 30), which is expected given the training images were likely higher quality. Brightness, contrast, rotation, and scaling perturbations show reasonable robustness (10-16% flip rates).

Full artifacts: `results/robustness/robustness_summary.csv`,
`results/robustness/robustness_results.csv`.

## Dataset Statistics

Run: `python scripts/dataset_stats.py --train-dir data/train --test-dir data/test`

| Split      | Total Images | Classes | Imbalance Ratio | Corrupted |
| ---------- | ----------- | ------- | --------------- | --------- |
| Train      | 13,898      | 22      | 6.7x            | 0         |
| Test       | 1,546       | 22      | 7.0x            | 0         |

The dataset is moderately imbalanced (6.7-7.0x ratio between largest and smallest classes). No corrupted images were found in either split.

Full artifacts: `results/dataset_stats.json`,
`results/class_distribution.png`.

## Performance Benchmarking

Run: `python scripts/benchmark.py`

| Metric                     | Value (CPU) |
| -------------------------- | ----------- |
| Model Load Time            | 1.01s       |
| Single-image Latency       | 127.27ms    |
| Batch 1 Throughput         | 8.6 img/s   |
| Batch 4 Throughput         | 10.4 img/s  |
| Batch 8 Throughput         | 11.0 img/s  |
| Batch 16 Throughput        | 10.5 img/s  |

Model verification shows 39,396,822 total parameters (150.3 MB fp32 size) with 22 classes and 224×224 input resolution.

Full artifacts: `results/benchmark.json`.

## Demo

Streamlit deployment:
https://skin-disease-classifier-fq9tu2ysuqwme9l9gds9xc.streamlit.app/

## Installation

```
git clone https://github.com/abhinavnagar29/Skin-disease-classifier.git
cd Skin-disease-classifier

python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

## Running Locally

```
streamlit run app.py
```

First run downloads the checkpoint from the Hugging Face Hub — needs
internet access once. `torch.cuda.is_available()` handles the CPU/GPU
switch automatically; no configuration needed either way.

## Evaluation Pipeline

```
python scripts/verify_model.py --checkpoint model.pt
python scripts/dataset_stats.py --train-dir data/train --test-dir data/test
python scripts/evaluate.py --test-dir data/test --checkpoint model.pt
python scripts/error_analysis.py
python scripts/calibration.py --test-dir data/test --checkpoint model.pt
python scripts/robustness.py --test-dir data/test --n-samples 100 --checkpoint model.pt
python scripts/benchmark.py --checkpoint model.pt
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
│   ├── verify_model.py      - sanity checks, parameter count, model size
│   ├── dataset_stats.py     - dataset statistics, class distribution, duplicate detection
│   ├── evaluate.py          - full test-set evaluation with metrics
│   ├── error_analysis.py    - built on evaluate.py's output
│   ├── calibration.py       - ECE + temperature scaling
│   ├── robustness.py        - robustness to perturbations
│   └── benchmark.py         - performance benchmarking (latency, throughput)
│
├── results/
│   ├── metrics.json
│   ├── classification_report.csv
│   ├── confusion_matrix.png / .npy
│   ├── per_class_f1.png
│   ├── confidence_distribution.png
│   ├── predictions.csv
│   ├── error_analysis/      - report.md + supporting CSVs
│   ├── calibration/         - ece_report.json, reliability_diagram.png
│   ├── robustness/          - robustness_results.csv, robustness_summary.csv
│   ├── dataset_stats.json
│   ├── class_distribution.png
│   └── benchmark.json
│
├── assets/                  - images/diagrams for this README
└── models/
    └── README.md            - why weights live on the HF Hub, not in git
```

## Limitations

- **Overconfidence:** raw ECE of 0.132, with mean confidence (88.7%)
  well above actual accuracy (75.5%) — raw softmax scores should not be
  read as calibrated probabilities without temperature scaling on a
  held-out validation set.
- **Weakest classes:** `Lupus`, `Moles`, `Infestations_Bites`,
  `Sun_Sunlight_Damage`, and `Vascular_Tumors` all sit at F1 < 0.66.
- **Systematic confusions:** `Benign_tumors` is a frequent false
  prediction for several other lesion classes, most notably absorbing
  `SkinCancer` cases (13 in this test set) — the most clinically
  significant failure mode found.
- **JPEG compression sensitivity:** 38% prediction flip rate at quality 30
  indicates the model may not generalize well to heavily compressed images
  from messaging apps or low-bandwidth scenarios.
- **Grad-CAM covers the ResNet50 branch only** — the DenseNet branch and
  the fused decision are not directly visualized.
- **No fairness/subgroup analysis** (e.g., across skin tones) has been
  performed — treat any deployment beyond research/demo use with real
  caution.
- **CPU-only benchmarking:** GPU performance numbers not available on this
  machine; actual deployment on GPU would show significantly higher
  throughput.

## Medical Disclaimer

This project is a research and educational demonstration only. It is not
a certified medical device, has not undergone clinical validation, and
predictions should never be used to make medical decisions. Always
consult a licensed dermatologist or physician for any real skin concern.
