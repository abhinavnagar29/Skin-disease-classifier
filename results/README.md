# Results

This directory is populated by running the scripts in `scripts/`. It is
intentionally empty in a fresh checkout — nothing here is committed as a
placeholder with fake numbers.

Run, in order:

```bash
python scripts/verify_model.py
python scripts/dataset_stats.py --train-dir data/train --val-dir data/val --test-dir data/test
python scripts/evaluate.py --test-dir data/test
python scripts/error_analysis.py
python scripts/robustness.py --test-dir data/test --n-samples 100
python scripts/calibration.py --test-dir data/test --val-dir data/val
python scripts/benchmark.py
```

After running these, this directory will contain:

```
results/
├── dataset_stats.json
├── class_distribution.png
├── metrics.json
├── classification_report.csv
├── confusion_matrix.png
├── confusion_matrix.npy
├── per_class_f1.png
├── confidence_distribution.png
├── predictions.csv
├── benchmark.json
├── error_analysis/
│   ├── report.md
│   ├── worst_classes.csv
│   ├── most_confused_pairs.csv
│   ├── high_confidence_errors.csv
│   └── low_confidence_correct.csv
├── robustness/
│   ├── robustness_results.csv
│   └── robustness_summary.csv
└── calibration/
    ├── ece_report.json
    ├── reliability_diagram.png
    └── temperature.json   (only if --val-dir was provided)
```

`app.py`'s "Model Performance" section reads `metrics.json` and
`confusion_matrix.png` directly — once you run `evaluate.py`, that
section of the app populates automatically, no code change needed.
