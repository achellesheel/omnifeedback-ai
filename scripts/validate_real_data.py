"""Validates V2 (BiLSTM) and V3 (DistilBERT) against real, independently-authored text
instead of the 8 hand-written stress-test sentences used in PROGRESS.md Milestone 8/9.

Proxy ground truth (neither dataset has a native "urgency" label):
  - Yelp Review Full: 5-class star rating (0=1-star ... 4=5-star). 1-star reviews are the
    proxy-critical group (analogous to a severe complaint), 5-star the proxy-low group.
    Also computes a Spearman correlation across all 5 classes for a more granular check.
  - Sentiment140: binary sentiment (0=negative, 1=positive). Negative is proxy-critical,
    positive is proxy-low.

Uses the same "critical-minus-low mean score separation" metric as the hand-written
stress test (scripts/stress_test_compare.py) so results are directly comparable — this
is a larger-n, real-text replication of that same measurement, not a new methodology.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestor import TextIngestor
from src.urgency import load_urgency_model, score_urgency
from src.variants import VARIANTS

N_PER_CLASS = 150  # per proxy-class sample size, per dataset — keeps CPU/MPS inference time reasonable
SEED = 42


def score_batch(bundle, texts):
    scores = []
    for t in texts:
        cleaned = TextIngestor.clean_text(t)
        scores.append(score_urgency(bundle, t, cleaned_text=cleaned))
    return scores


def validate_yelp(v2_bundle, v3_bundle, path):
    df = pd.read_csv(path)
    rng = np.random.RandomState(SEED)

    critical = df[df["label"] == 0].sample(n=min(N_PER_CLASS, (df["label"] == 0).sum()), random_state=SEED)
    low = df[df["label"] == 4].sample(n=min(N_PER_CLASS, (df["label"] == 4).sum()), random_state=SEED)

    v2_crit = score_batch(v2_bundle, critical["text"].tolist())
    v2_low = score_batch(v2_bundle, low["text"].tolist())
    v3_crit = score_batch(v3_bundle, critical["text"].tolist())
    v3_low = score_batch(v3_bundle, low["text"].tolist())

    # Spearman correlation across all 5 star classes (more granular than the 2-group split)
    all_sample = df.groupby("label", group_keys=False)[df.columns].apply(
        lambda g: g.sample(n=min(80, len(g)), random_state=SEED)
    )
    v2_all = score_batch(v2_bundle, all_sample["text"].tolist())
    v3_all = score_batch(v3_bundle, all_sample["text"].tolist())
    # star rating is "higher = better"; urgency is "higher = worse", so we correlate against
    # inverted rating (4 - label) so a positive correlation means "higher urgency = worse review"
    inverted_rating = (4 - all_sample["label"]).tolist()
    v2_corr, v2_p = spearmanr(inverted_rating, v2_all)
    v3_corr, v3_p = spearmanr(inverted_rating, v3_all)

    v2_u, v2_u_p = mannwhitneyu(v2_crit, v2_low, alternative="greater")
    v3_u, v3_u_p = mannwhitneyu(v3_crit, v3_low, alternative="greater")

    return {
        "dataset": "Yelp Review Full",
        "n_per_class": len(critical),
        "v2_critical_mean": round(float(np.mean(v2_crit)), 4),
        "v2_low_mean": round(float(np.mean(v2_low)), 4),
        "v2_separation": round(float(np.mean(v2_crit) - np.mean(v2_low)), 4),
        "v2_mannwhitney_p": round(float(v2_u_p), 6),
        "v3_critical_mean": round(float(np.mean(v3_crit)), 4),
        "v3_low_mean": round(float(np.mean(v3_low)), 4),
        "v3_separation": round(float(np.mean(v3_crit) - np.mean(v3_low)), 4),
        "v3_mannwhitney_p": round(float(v3_u_p), 6),
        "v2_spearman_vs_5class_rating": {"rho": round(float(v2_corr), 4), "p": round(float(v2_p), 6), "n": len(all_sample)},
        "v3_spearman_vs_5class_rating": {"rho": round(float(v3_corr), 4), "p": round(float(v3_p), 6), "n": len(all_sample)},
    }


def validate_sentiment140(v2_bundle, v3_bundle, path):
    df = pd.read_csv(path)

    critical = df[df["label"] == 0].sample(n=min(N_PER_CLASS, (df["label"] == 0).sum()), random_state=SEED)
    low = df[df["label"] == 1].sample(n=min(N_PER_CLASS, (df["label"] == 1).sum()), random_state=SEED)

    v2_crit = score_batch(v2_bundle, critical["text"].tolist())
    v2_low = score_batch(v2_bundle, low["text"].tolist())
    v3_crit = score_batch(v3_bundle, critical["text"].tolist())
    v3_low = score_batch(v3_bundle, low["text"].tolist())

    v2_u, v2_u_p = mannwhitneyu(v2_crit, v2_low, alternative="greater")
    v3_u, v3_u_p = mannwhitneyu(v3_crit, v3_low, alternative="greater")

    return {
        "dataset": "Sentiment140",
        "n_per_class": len(critical),
        "v2_critical_mean": round(float(np.mean(v2_crit)), 4),
        "v2_low_mean": round(float(np.mean(v2_low)), 4),
        "v2_separation": round(float(np.mean(v2_crit) - np.mean(v2_low)), 4),
        "v2_mannwhitney_p": round(float(v2_u_p), 6),
        "v3_critical_mean": round(float(np.mean(v3_crit)), 4),
        "v3_low_mean": round(float(np.mean(v3_low)), 4),
        "v3_separation": round(float(np.mean(v3_crit) - np.mean(v3_low)), 4),
        "v3_mannwhitney_p": round(float(v3_u_p), 6),
    }


def main():
    v2_bundle = load_urgency_model(VARIANTS["v2_hardened"])
    v3_bundle = load_urgency_model(VARIANTS["v3_transformer"])

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    results = {
        "yelp": validate_yelp(v2_bundle, v3_bundle, os.path.join(raw_dir, "yelp_reviews_sample.csv")),
        "sentiment140": validate_sentiment140(v2_bundle, v3_bundle, os.path.join(raw_dir, "sentiment140_sample.csv")),
    }

    for name, r in results.items():
        print(f"\n=== {r['dataset']} (n={r['n_per_class']}/class) ===")
        print(f"V2: critical_mean={r['v2_critical_mean']} low_mean={r['v2_low_mean']} separation={r['v2_separation']}")
        print(f"V3: critical_mean={r['v3_critical_mean']} low_mean={r['v3_low_mean']} separation={r['v3_separation']}")
        if "v2_spearman_vs_5class_rating" in r:
            print(f"V2 Spearman vs 5-class rating: {r['v2_spearman_vs_5class_rating']}")
            print(f"V3 Spearman vs 5-class rating: {r['v3_spearman_vs_5class_rating']}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "models", "real_data_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
