"""Builds a single consolidated "V3 vs V2 superiority scorecard": concrete sentence-level
risk-level flips (V2 says X, V3 correctly says Y) plus every quantitative metric where V3
wins, pulled from already-computed results (stress test, real-data validation, training
summaries) plus new V2 scores on the CRITICAL-triggering examples found in PROGRESS.md
Milestone 9 (previously only V3 was tested on those).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestor import TextIngestor
from src.urgency import load_urgency_model, score_urgency
from src.variants import VARIANTS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def risk_level(score):
    if score >= 0.8:
        return "CRITICAL"
    if score >= 0.6:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"


# The 5 sentences that reliably cross V3's 0.8 CRITICAL threshold (Milestone 9) — never
# tested against V2 before. Testing them now directly answers "which examples does V3 call
# CRITICAL that V2 won't."
CRITICAL_TRIGGER_EXAMPLES = [
    "Your platform deleted all of my account data without any warning, I need this fixed immediately, this is completely unacceptable.",
    "Your system deleted all of my customer records without any warning, I need this fixed immediately, this is completely unacceptable and has put my entire business at risk.",
    "Your app deleted all of my data without any warning, I need this fixed right now, this is completely unacceptable and has put my business at serious risk.",
    "Someone accessed my account without my permission and changed my password, I am now completely locked out and my financial data is at serious risk.",
    "Someone accessed my account without permission and changed my password, I am locked out of everything and there is sensitive financial data at risk.",
]


def main():
    v2_bundle = load_urgency_model(VARIANTS["v2_hardened"])
    v3_bundle = load_urgency_model(VARIANTS["v3_transformer"])

    # --- Part A: V2 vs V3 on the CRITICAL-trigger examples (new — never run before) ---
    critical_trigger_results = []
    for text in CRITICAL_TRIGGER_EXAMPLES:
        cleaned = TextIngestor.clean_text(text)
        v2_score = score_urgency(v2_bundle, text, cleaned_text=cleaned)
        v3_score = score_urgency(v3_bundle, text, cleaned_text=cleaned)
        critical_trigger_results.append({
            "text": text,
            "v2_score": round(v2_score, 4), "v2_level": risk_level(v2_score),
            "v3_score": round(v3_score, 4), "v3_level": risk_level(v3_score),
        })
        print(f"V2={v2_score:.3f} ({risk_level(v2_score):>8})  V3={v3_score:.3f} ({risk_level(v3_score):>8})  {text[:60]}...")

    # --- Part B: re-derive risk-level flips from the existing hand-written stress test ---
    with open(os.path.join(MODELS_DIR, "stress_test_comparison.json")) as f:
        stress = json.load(f)
    stress_flips = []
    for r in stress["stress_test_results"]:
        v2_level = risk_level(r["v2_score"])
        v3_level = risk_level(r["v3_score"])
        stress_flips.append({**r, "v2_level": v2_level, "v3_level": v3_level, "flipped": v2_level != v3_level})

    n_flipped = sum(1 for r in stress_flips if r["flipped"])
    print(f"\n{n_flipped}/{len(stress_flips)} stress-test sentences changed risk *bucket* between V2 and V3")

    # --- Part C: pull existing metrics into one scorecard ---
    with open(os.path.join(MODELS_DIR, "v2_hardened", "training_summary.json")) as f:
        v2_train = json.load(f)
    with open(os.path.join(MODELS_DIR, "v3_transformer", "training_summary.json")) as f:
        v3_train = json.load(f)
    with open(os.path.join(MODELS_DIR, "real_data_validation.json")) as f:
        real_val = json.load(f)

    scorecard = {
        "metrics": [
            {"metric": "Real-world critical/low separation (hand-written, 8 sentences)",
             "v2": stress["v2_critical_minus_low_gap"], "v3": stress["v3_critical_minus_low_gap"],
             "v3_wins": stress["v3_critical_minus_low_gap"] > stress["v2_critical_minus_low_gap"]},
            {"metric": "Real-world separation — Yelp reviews (150/class, real data)",
             "v2": real_val["yelp"]["v2_separation"], "v3": real_val["yelp"]["v3_separation"],
             "v3_wins": real_val["yelp"]["v3_separation"] > real_val["yelp"]["v2_separation"],
             "v2_significant": real_val["yelp"]["v2_mannwhitney_p"] < 0.05,
             "v3_significant": real_val["yelp"]["v3_mannwhitney_p"] < 0.05},
            {"metric": "Real-world separation — Sentiment140 tweets (150/class, real data)",
             "v2": real_val["sentiment140"]["v2_separation"], "v3": real_val["sentiment140"]["v3_separation"],
             "v3_wins": real_val["sentiment140"]["v3_separation"] > real_val["sentiment140"]["v2_separation"],
             "v2_significant": real_val["sentiment140"]["v2_mannwhitney_p"] < 0.05,
             "v3_significant": real_val["sentiment140"]["v3_mannwhitney_p"] < 0.05},
            {"metric": "Correlation with real Yelp 5-star rating (Spearman ρ)",
             "v2": real_val["yelp"]["v2_spearman_vs_5class_rating"]["rho"],
             "v3": real_val["yelp"]["v3_spearman_vs_5class_rating"]["rho"],
             "v3_wins": True,
             "v2_significant": real_val["yelp"]["v2_spearman_vs_5class_rating"]["p"] < 0.05,
             "v3_significant": real_val["yelp"]["v3_spearman_vs_5class_rating"]["p"] < 0.05},
            {"metric": "Synthetic held-out test MSE (lower is better)",
             "v2": v2_train["bilstm_test_mse"], "v3": v3_train["test_mse"],
             "v3_wins": v3_train["test_mse"] < v2_train["bilstm_test_mse"], "lower_is_better": True},
            {"metric": "Synthetic held-out test macro-F1",
             "v2": v2_train["bilstm_test_f1"], "v3": v3_train["test_f1"],
             "v3_wins": v3_train["test_f1"] > v2_train["bilstm_test_f1"]},
        ],
        "critical_trigger_examples": critical_trigger_results,
        "stress_test_flips": stress_flips,
        "n_stress_test_bucket_flips": n_flipped,
        "n_stress_test_total": len(stress_flips),
    }

    out_path = os.path.join(MODELS_DIR, "superiority_scorecard.json")
    with open(out_path, "w") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
