"""
Phase 4 — error analysis, built on top of evaluate.py's output.

Requires results/predictions.csv and results/confusion_matrix.npy to
already exist — run scripts/evaluate.py first.

Answers, with actual numbers:
    - worst-performing classes (lowest F1 / recall)
    - most confused class pairs (from the confusion matrix, off-diagonal)
    - high-confidence wrong predictions (the model was confidently wrong —
      most interesting failure mode to discuss in an interview)
    - low-confidence correct predictions (right, but not sure why it should be)

Outputs under results/error_analysis/:
    report.md
    worst_classes.csv
    most_confused_pairs.csv
    high_confidence_errors.csv
    low_confidence_correct.csv

Run:
    python scripts/error_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")


def main():
    results_dir = "results"
    out_dir = f"{results_dir}/error_analysis"
    os.makedirs(out_dir, exist_ok=True)

    predictions_path = f"{results_dir}/predictions.csv"
    report_path = f"{results_dir}/classification_report.csv"
    cm_path = f"{results_dir}/confusion_matrix.npy"

    for p in [predictions_path, report_path, cm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found — run scripts/evaluate.py first, error "
                f"analysis is derived from its output, not independent."
            )

    preds = pd.read_csv(predictions_path)
    report = pd.read_csv(report_path, index_col=0)
    cm = np.load(cm_path)

    classes = [c for c in report.index if c not in ("accuracy", "macro avg", "weighted avg")]

    # ---- worst-performing classes ----
    class_report = report.loc[classes].copy()
    class_report = class_report.sort_values("f1-score")
    class_report.to_csv(f"{out_dir}/worst_classes.csv")

    worst_5 = class_report.head(5)

    # ---- most confused pairs (off-diagonal confusion matrix entries) ----
    pairs = []
    for i, true_c in enumerate(classes):
        for j, pred_c in enumerate(classes):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_class": true_c,
                    "predicted_as": pred_c,
                    "count": int(cm[i, j]),
                })
    pairs_df = pd.DataFrame(pairs).sort_values("count", ascending=False)
    pairs_df.to_csv(f"{out_dir}/most_confused_pairs.csv", index=False)

    # ---- high-confidence wrong predictions ----
    wrong = preds[~preds["correct"]].copy()
    high_conf_errors = wrong.sort_values("confidence", ascending=False).head(20)
    high_conf_errors.to_csv(f"{out_dir}/high_confidence_errors.csv", index=False)

    # ---- low-confidence correct predictions ----
    correct = preds[preds["correct"]].copy()
    low_conf_correct = correct.sort_values("confidence", ascending=True).head(20)
    low_conf_correct.to_csv(f"{out_dir}/low_confidence_correct.csv", index=False)

    # ---- markdown report ----
    with open(f"{out_dir}/report.md", "w") as f:
        f.write("# Error Analysis\n\n")

        f.write("## Worst-performing classes (by F1)\n\n")
        f.write(worst_5[["precision", "recall", "f1-score", "support"]].to_markdown())
        f.write("\n\n")

        f.write("## Most confused class pairs\n\n")
        f.write(pairs_df.head(10).to_markdown(index=False))
        f.write("\n\n")

        f.write("## High-confidence wrong predictions\n\n")
        f.write(
            f"{len(wrong)} total wrong predictions. "
            f"Top 20 by confidence shown in `high_confidence_errors.csv`. "
            f"Mean confidence on wrong predictions: {wrong['confidence'].mean():.3f}\n\n"
        )

        f.write("## Low-confidence correct predictions\n\n")
        f.write(
            f"Mean confidence on correct predictions: {correct['confidence'].mean():.3f}. "
            f"Bottom 20 by confidence shown in `low_confidence_correct.csv`.\n\n"
        )

        f.write("## Summary\n\n")
        f.write(f"- Total predictions: {len(preds)}\n")
        f.write(f"- Correct: {len(correct)} ({len(correct)/len(preds)*100:.1f}%)\n")
        f.write(f"- Wrong: {len(wrong)} ({len(wrong)/len(preds)*100:.1f}%)\n")
        f.write(f"- Worst class by F1: {worst_5.index[0]} (F1={worst_5.iloc[0]['f1-score']:.3f})\n")
        if len(pairs_df) > 0:
            top_pair = pairs_df.iloc[0]
            f.write(
                f"- Most confused pair: '{top_pair['true_class']}' predicted as "
                f"'{top_pair['predicted_as']}' ({top_pair['count']} times)\n"
            )

    print(f"Error analysis written to {out_dir}/")
    print(f"\nWorst classes:\n{worst_5[['precision', 'recall', 'f1-score']]}")
    print(f"\nTop confused pairs:\n{pairs_df.head(5)}")


if __name__ == "__main__":
    main()
