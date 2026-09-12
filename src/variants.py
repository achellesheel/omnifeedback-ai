"""Registry of dataset/model variants so the live app can toggle between them.

V1 ("naive") reproduces the original capstone generator: verbatim phrases picked from a
tiny fixed list per class, no ambiguous cases, no label noise — trivially separable text.
V2 ("hardened") is the production version: templated + randomized phrasing, deliberately
ambiguous mixed-sentiment cases, and ~3% label noise. Keeping both trained and queryable
side by side turns "we fixed an overfitting-prone dataset" from a claim in PROGRESS.md into
something a viewer can click and see for themselves.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "v1_naive": {
        "label": "V1 · Naive (overfit-prone)",
        "short_label": "V1 Naive",
        "description": (
            "Original capstone-style generator: verbatim phrases from a small fixed list per "
            "class, no ambiguous cases, no label noise. Text is trivially separable by exact "
            "keyword overlap — metrics look perfect but don't reflect real-world difficulty."
        ),
        "raw_csv": ROOT / "data" / "raw" / "omnifeedback_v1_naive.csv",
        "db_path": ROOT / "data" / "warehouse" / "omnifeedback_v1_naive.db",
        "model_dir": ROOT / "models" / "v1_naive",
    },
    "v2_hardened": {
        "label": "V2 · Hardened (production)",
        "short_label": "V2 Hardened",
        "description": (
            "Templated + randomized phrasing, deliberately ambiguous mixed-sentiment cases "
            "straddling the urgency threshold, and ~3% simulated label noise. Metrics are "
            "lower but credible — this is the version the production system ships with."
        ),
        "raw_csv": ROOT / "data" / "raw" / "omnifeedback_v2_hardened.csv",
        "db_path": ROOT / "data" / "warehouse" / "omnifeedback_v2_hardened.db",
        "model_dir": ROOT / "models" / "v2_hardened",
    },
}

DEFAULT_VARIANT = "v2_hardened"
