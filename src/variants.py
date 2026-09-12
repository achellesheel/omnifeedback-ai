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
        "model_type": "bilstm",
    },
    "v2_hardened": {
        "label": "V2 · Hardened (production)",
        "short_label": "V2 Hardened",
        "description": (
            "Templated + randomized phrasing, deliberately ambiguous mixed-sentiment cases "
            "straddling the urgency threshold, and ~3% simulated label noise. Metrics are "
            "lower but credible — but its from-scratch 196-word vocabulary still fails on "
            "real-world phrasing outside its training templates (see V3)."
        ),
        "raw_csv": ROOT / "data" / "raw" / "omnifeedback_v2_hardened.csv",
        "db_path": ROOT / "data" / "warehouse" / "omnifeedback_v2_hardened.db",
        "model_dir": ROOT / "models" / "v2_hardened",
        "model_type": "bilstm",
    },
    "v3_transformer": {
        "label": "V3 · Transfer-Learned (DistilBERT)",
        "short_label": "V3 Transformer",
        "description": (
            "Same hardened dataset as V2, but the urgency regressor is now a fine-tuned "
            "DistilBERT (frozen early layers, trainable last 2 layers + head) instead of a "
            "from-scratch BiLSTM. Subword tokenization means no closed-vocabulary OOV problem "
            "— it generalizes to real customer phrasing V2 couldn't handle."
        ),
        "raw_csv": ROOT / "data" / "raw" / "omnifeedback_v2_hardened.csv",
        "db_path": ROOT / "data" / "warehouse" / "omnifeedback_v2_hardened.db",
        "model_dir": ROOT / "models" / "v3_transformer",
        "model_type": "transformer",
    },
}

DEFAULT_VARIANT = "v3_transformer"
