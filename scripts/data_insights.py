"""Computes real data-insight metrics from the trained V2 (production) model for the
Step 6 analysis: confusion matrix, threshold sensitivity, per-channel/aspect urgency
breakdown, and misclassification examples. Writes models/v2_hardened/data_insights.json
so the numbers in docs/DATA_INSIGHTS.md are traceable to an actual run, not invented.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.database import WarehouseManager
from src.dl_lstm import load_bilstm_model, predict_urgency
from src.ingestor import TextIngestor
from src.variants import VARIANTS

VARIANT = "v2_hardened"


def main():
    variant = VARIANTS[VARIANT]
    db = WarehouseManager(db_path=variant["db_path"])
    df = db.fetch_all_feedback()
    model, vocab, device = load_bilstm_model(model_dir=variant["model_dir"])

    df["pred_urgency"] = df["cleaned_text"].apply(lambda t: predict_urgency(model, vocab, t, device))
    df["true_critical"] = (df["urgency_score"] >= 0.5).astype(int)
    df["pred_critical_05"] = (df["pred_urgency"] >= 0.5).astype(int)

    # --- Confusion matrix at the default 0.5 threshold ---
    cm = confusion_matrix(df["true_critical"], df["pred_critical_05"])
    tn, fp, fn, tp = cm.ravel()

    # --- Threshold sensitivity sweep: find the macro-F1-optimal threshold ---
    # NOTE: sklearn's precision_recall_curve only tracks the *positive-class* F1, which is not
    # comparable to the macro-F1 reported elsewhere in this project (avg across both classes) —
    # first pass at this script conflated the two and produced a nonsensical "best" threshold
    # that scored *worse* than the 0.5 default. Fixed by sweeping macro-F1 directly so the
    # comparison is apples-to-apples.
    candidate_thresholds = np.linspace(0.05, 0.95, 181)
    macro_f1s = [
        f1_score(df["true_critical"], (df["pred_urgency"] >= t).astype(int), average="macro")
        for t in candidate_thresholds
    ]
    best_idx = int(np.argmax(macro_f1s))
    best_threshold = float(candidate_thresholds[best_idx])
    best_f1 = float(macro_f1s[best_idx])
    f1_at_05 = float(f1_score(df["true_critical"], df["pred_critical_05"], average="macro"))

    df["pred_critical_best"] = (df["pred_urgency"] >= best_threshold).astype(int)
    cm_best = confusion_matrix(df["true_critical"], df["pred_critical_best"])
    tn_b, fp_b, fn_b, tp_b = cm_best.ravel()

    # --- Correlation between predicted and true urgency (continuous calibration check) ---
    correlation = float(np.corrcoef(df["urgency_score"], df["pred_urgency"])[0, 1])

    # --- Per-channel / per-aspect breakdown ---
    channel_breakdown = df.groupby("channel")["urgency_score"].agg(["mean", "count"]).round(4).to_dict("index")
    aspect_breakdown = df.groupby("aspect_category")["urgency_score"].agg(["mean", "count"]).round(4).to_dict("index")

    # --- Chi-squared: channel vs urgency bucket (reproducing the app's stats page numerically) ---
    def bucket(s):
        return "Critical" if s >= 0.65 else ("Medium" if s >= 0.35 else "Low")
    df["urgency_bucket"] = df["urgency_score"].apply(bucket)
    contingency = pd.crosstab(df["channel"], df["urgency_bucket"])
    chi2, p_value, dof, _ = chi2_contingency(contingency)

    # --- Misclassification examples (false negatives: missed critical feedback) ---
    false_negatives = df[(df["true_critical"] == 1) & (df["pred_critical_05"] == 0)]
    false_positives = df[(df["true_critical"] == 0) & (df["pred_critical_05"] == 1)]

    results = {
        "variant": VARIANT,
        "n_records": len(df),
        "confusion_matrix_at_0.5": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "f1_at_0.5": f1_at_05,
        "best_threshold": round(best_threshold, 4),
        "f1_at_best_threshold": round(best_f1, 4),
        "threshold_sweep": {
            "thresholds": [round(float(t), 3) for t in candidate_thresholds[::6]],
            "macro_f1": [round(float(v), 4) for v in macro_f1s[::6]],
        },
        "confusion_matrix_at_best_threshold": {"tn": int(tn_b), "fp": int(fp_b), "fn": int(fn_b), "tp": int(tp_b)},
        "pred_vs_true_correlation": round(correlation, 4),
        "channel_breakdown": channel_breakdown,
        "aspect_breakdown": aspect_breakdown,
        "chi2_channel_vs_urgency": {"chi2": round(float(chi2), 4), "p_value": round(float(p_value), 6), "dof": int(dof)},
        "n_false_negatives": len(false_negatives),
        "n_false_positives": len(false_positives),
        "false_negative_examples": false_negatives["raw_text"].head(5).tolist(),
        "false_positive_examples": false_positives["raw_text"].head(5).tolist(),
    }

    out_path = variant["model_dir"] / "data_insights.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
