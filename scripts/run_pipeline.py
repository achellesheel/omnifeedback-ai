"""End-to-end pipeline: clean text -> load warehouse -> train classical ML -> train BiLSTM.

Run after scripts/generate_data.py, once per variant:
    python scripts/run_pipeline.py v1_naive
    python scripts/run_pipeline.py v2_hardened

Prints metrics that get hand-copied into PROGRESS.md so results are auditable rather
than just trusted, and writes models/<variant>/training_summary.json for the app's
live Model Comparison page.
"""
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.database import WarehouseManager
from src.dl_lstm import train_bilstm_model
from src.ingestor import TextIngestor
from src.ml_models import run_ml_pipeline
from src.variants import DEFAULT_VARIANT, VARIANTS


def run(variant_id: str):
    variant = VARIANTS[variant_id]
    t0 = time.time()
    df = pd.read_csv(variant["raw_csv"])
    print(f"[{variant_id}] Loaded {len(df)} raw records")

    df["cleaned_text"] = df["raw_text"].apply(TextIngestor.clean_text)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    print(f"[{variant_id}] {len(df)} records remain after cleaning (dropped empty)")

    db = WarehouseManager(db_path=variant["db_path"])
    inserted = db.insert_feedback_batch(df)
    print(f"[{variant_id}] Inserted {inserted} rows into warehouse (ACID transaction)")

    ingest_latency_ms = (time.time() - t0) * 1000 / len(df)
    print(f"[{variant_id}] Avg ingestion latency: {ingest_latency_ms:.3f} ms/record")

    print(f"\n=== [{variant_id}] Classical ML pipeline ===")
    ml_results = run_ml_pipeline(df, model_dir=variant["model_dir"])
    print(f"Best K (Kneedle elbow): {ml_results['best_k']}")
    for k, s in ml_results["k_sweep_scores"].items():
        print(f"  k={k}: silhouette={s['silhouette']:.4f} inertia={s['inertia']:.1f}")
    for name, res in ml_results["baseline_results"].items():
        print(f"  {name}: macro-F1={res['f1']:.4f}")

    print(f"\n=== [{variant_id}] BiLSTM urgency regressor ===")
    dl_results = train_bilstm_model(df, epochs=30, patience=4, model_dir=variant["model_dir"])
    print(f"Stopped at epoch {dl_results['stopped_epoch']} (early stopping)")
    print(f"Best val MSE: {dl_results['best_val_loss']:.5f}")
    print(f"Test MSE: {dl_results['test_mse']:.5f}")
    print(f"Test macro-F1 (thresholded @0.5): {dl_results['test_f1']:.5f}")

    summary = {
        "variant_id": variant_id,
        "variant_label": variant["label"],
        "n_records": len(df),
        "ingest_latency_ms_per_record": ingest_latency_ms,
        "best_k": ml_results["best_k"],
        "k_sweep_scores": ml_results["k_sweep_scores"],
        "baseline_f1": {n: r["f1"] for n, r in ml_results["baseline_results"].items()},
        "bilstm_stopped_epoch": dl_results["stopped_epoch"],
        "bilstm_best_val_mse": dl_results["best_val_loss"],
        "bilstm_test_mse": dl_results["test_mse"],
        "bilstm_test_f1": dl_results["test_f1"],
        "bilstm_history": dl_results["history"],
    }
    out_path = variant["model_dir"] / "training_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{variant_id}] Saved training summary -> {out_path}")
    print(f"[{variant_id}] Total pipeline time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    variants_to_run = sys.argv[1:] or [DEFAULT_VARIANT]
    for v in variants_to_run:
        if v not in VARIANTS:
            raise SystemExit(f"Unknown variant '{v}'. Choices: {list(VARIANTS)}")
        run(v)
        print("\n" + "=" * 70 + "\n")
