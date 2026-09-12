"""Runs the same out-of-distribution stress-test sentences through both V2
(BiLSTM) and V3 (DistilBERT) and saves a side-by-side comparison for the app's
Model Comparison page. These sentences are deliberately NOT phrased like the
synthetic training templates — they're what exposed V2's generalization
failure in the first place (see PROGRESS.md Milestone 8).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestor import TextIngestor
from src.urgency import load_urgency_model, score_urgency
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
    v2_bundle = load_urgency_model(VARIANTS["v2_hardened"])
    v3_bundle = load_urgency_model(VARIANTS["v3_transformer"])

    results = []
    for expected, text in STRESS_TEST:
        cleaned = TextIngestor.clean_text(text)
        v2_score = score_urgency(v2_bundle, text, cleaned_text=cleaned)
        v3_score = score_urgency(v3_bundle, text, cleaned_text=cleaned)
        results.append({
            "expected": expected, "text": text,
            "v2_score": round(v2_score, 4), "v3_score": round(v3_score, 4),
        })
        print(f"[{expected:>9}] V2={v2_score:.3f}  V3={v3_score:.3f}  {text[:60]}...")

    # A crude but concrete separation metric: mean(critical scores) - mean(low scores).
    # V2 should show ~0 separation (its whole failure mode); V3 should show a large gap.
    def sep(key):
        crit = [r[key] for r in results if r["expected"] == "CRITICAL"]
        low = [r[key] for r in results if r["expected"] == "LOW"]
        return round(sum(crit) / len(crit) - sum(low) / len(low), 4)

    summary = {
        "stress_test_results": results,
        "v2_critical_minus_low_gap": sep("v2_score"),
        "v3_critical_minus_low_gap": sep("v3_score"),
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "models", "stress_test_comparison.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nV2 critical-vs-low separation: {summary['v2_critical_minus_low_gap']}")
    print(f"V3 critical-vs-low separation: {summary['v3_critical_minus_low_gap']}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
