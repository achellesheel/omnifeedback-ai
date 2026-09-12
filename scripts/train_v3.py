"""Trains V3: a transfer-learned DistilBERT urgency regressor on the same V2
(hardened) dataset, then immediately re-runs the exact real-world stress-test
sentences that exposed V2's generalization failure, so the fix is verified
against the actual failure, not just against held-out synthetic test rows.
"""
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestor import TextIngestor
from src.transformer_regressor import predict_urgency_transformer, train_transformer_regressor
from src.variants import VARIANTS

STRESS_TEST = [
    ("CRITICAL", "Our production checkout API has been down for 45 minutes and we're losing real revenue every minute — nobody from support has called us back."),
    ("CRITICAL", "I was charged three times for the same subscription this month and your chatbot keeps looping me back to the same FAQ page."),
    ("CRITICAL", "The mobile app crashes every single time I try to check out, on three different phones. I've already switched to a competitor."),
    ("LOW", "Just wanted to say the redesigned onboarding flow is fantastic — set up my whole team in under 10 minutes."),
    ("LOW", "Support resolved my billing question in one chat message. Really impressed."),
    ("AMBIGUOUS", "Loved the new UI, but it logged me out twice today for no reason."),
    ("AMBIGUOUS", "Mostly happy with the service, though the search feature has been a bit flaky since the update."),
    ("AMBIGUOUS", "Shipping has been slower than promised — order placed 9 days ago, tracking hasn't updated in 4 days."),
]


def main():
    t0 = time.time()
    variant = VARIANTS["v2_hardened"]
    df = pd.read_csv(variant["raw_csv"])
    df["cleaned_text"] = df["raw_text"].apply(TextIngestor.clean_text)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    print(f"Training on {len(df)} rows (same data as V2 Hardened)")

    v3_model_dir = VARIANTS["v3_transformer"]["model_dir"]
    result = train_transformer_regressor(df, epochs=8, patience=3, model_dir=v3_model_dir)

    print(f"\nStopped at epoch {result['stopped_epoch']}")
    print(f"Best val MSE: {result['best_val_loss']:.5f}")
    print(f"Test MSE: {result['test_mse']:.5f}")
    print(f"Test macro-F1: {result['test_f1']:.5f}")
    print(f"Trainable params: {result['trainable_fraction']*100:.1f}%")

    print("\n=== Stress test: same sentences that broke V2 ===")
    stress_results = []
    for label, text in STRESS_TEST:
        score = predict_urgency_transformer(result["model"], result["tokenizer"], text, "mps" if __import__("torch").backends.mps.is_available() else "cpu")
        stress_results.append({"expected": label, "text": text, "v3_score": round(score, 4)})
        print(f"{score:.3f}  [{label}]  {text[:70]}...")

    summary = {
        "variant_id": "v3_transformer",
        "base_model": "distilbert-base-uncased",
        "n_records": len(df),
        "stopped_epoch": result["stopped_epoch"],
        "best_val_mse": result["best_val_loss"],
        "test_mse": result["test_mse"],
        "test_f1": result["test_f1"],
        "trainable_param_fraction": result["trainable_fraction"],
        "history": result["history"],
        "stress_test_results": stress_results,
    }
    out_path = v3_model_dir / "training_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
