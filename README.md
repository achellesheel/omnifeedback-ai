# OmniFeedback AI

**An enterprise customer-feedback intelligence platform: ingestion, escalation triage, and a GenAI resolution copilot — with the overfitting fix made visible, live.**

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.3x-red.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/tests-31%20passing-green.svg)](./tests)

Most feedback-analytics demos show you one model and ask you to trust the metric. This one shows you
**three** — the same pipeline evolved through two real, publicly-documented bugs: an overfitting-prone
dataset (V1 → V2), and a generalization failure fixed with transfer learning (V2 → V3, after a real
person testing the live app found it scoring obviously-critical and obviously-positive feedback almost
identically). Flip between all three in the app and watch the difference happen live.

Under the hood: multi-channel feedback ingestion through a regex cleaner into a transactional SQL star
schema, unsupervised K-Means aspect clustering (with a proper Kneedle-elbow k-selection, not a naive
argmax), an urgency regressor available in two forms — a from-scratch PyTorch BiLSTM and a transfer-learned
DistilBERT — HuggingFace BERT NER + BART executive summarization, and a Chain-of-Thought GenAI resolution
copilot with a pluggable backend that needs zero API keys to run.

📓 See [`PROGRESS.md`](./PROGRESS.md) for the full build log — every bug hit, every fix, and the
reasoning behind each modeling choice, written as it happened.

📊 [15-slide technical walkthrough](./docs/slides/index.html) · 📈 [Data insights](./docs/DATA_INSIGHTS.md)
· 🔍 [Product review](./docs/PRODUCT_REVIEW.md) · 🚀 [Enhancement roadmap](./docs/ENHANCEMENTS.md)

---

## Live Demo Feature: V1 → V2 → V3

Open the **Model Comparison** page and flip between three variants:
- **V1 Naive** — verbatim phrase lists, ~9 unique training sentences. Silhouette = 1.0, macro-F1 = 1.0.
  Looks perfect. Isn't — the model memorized 9 sentences.
- **V2 Hardened** — templated/randomized text, ambiguous mixed-sentiment cases, simulated label noise.
  Silhouette = 0.33, macro-F1 = 0.91/0.82. Lower numbers, real generalization on synthetic data — but its
  from-scratch, 196-word vocabulary still collapses on ordinary real-world phrasing (see below).
- **V3 Transfer-Learned** — same data as V2, but the urgency regressor is a fine-tuned DistilBERT (frozen
  early layers, ~21% of parameters trainable) instead of a from-scratch BiLSTM. On 8 natural-language
  stress-test sentences that V2 scored almost identically regardless of severity (critical vs. positive
  separation of just 0.02), V3 separates them by 0.30 — roughly 13x larger, using no more training data.

Every other page (Inference Playground, Clustering Explorer, GenAI Copilot, Statistical Tests) has the
same sidebar toggle, so you can watch clusters, urgency scores, and chi-squared results shift with the
underlying data or model in real time.

---

## Architecture

```
Raw feedback (CSV) ──▶ Regex TextIngestor ──▶ SQLite Star Schema Warehouse (ACID transactions)
                                                              │
        ┌─────────────────────────────┬────────────────────────────┬──────────────────────┐
        ▼                             ▼                            ▼                      │
  TF-IDF + KMeans              Urgency Regressor             BERT NER +               (all read from
  aspect clustering            BiLSTM (V1/V2) or             BART summarizer           the warehouse)
  (Kneedle elbow)              transfer-learned                                          │
  + LogReg/RF baseline         DistilBERT (V3)                                            │
        └─────────────────────────────┴────────────────────────────┴──────────────────────┘
                                              │
                                    GenAI Resolution Copilot
                                (CoT + few-shot, pluggable backend:
                                 local rules ↔ Claude API)
                                              │
                                  Streamlit Multi-Page Dashboard
                                  (V1 / V2 / V3 toggle on every page)
```

---

## Repository Structure

```
├── app.py                        # Streamlit entrypoint: Executive Overview
├── pages/
│   ├── 0_Model_Comparison.py      # V1 vs. V2 vs. V3 side by side — the headline feature
│   ├── 1_Inference_Playground.py  # Live urgency scoring (either model backend)
│   ├── 2_Clustering_Explorer.py   # K-Means clusters + elbow/silhouette justification
│   ├── 3_GenAI_Copilot.py         # CoT/few-shot resolution drafting
│   ├── 4_NER_and_Summarization.py # BERT NER + BART executive briefing
│   └── 5_Statistical_Tests.py     # Chi-squared channel/urgency independence test
├── src/
│   ├── variants.py                # V1/V2/V3 registry (paths, labels, descriptions, model_type)
│   ├── app_state.py                # Shared sidebar variant selector
│   ├── urgency.py                  # Unified dispatcher: routes to BiLSTM or transformer backend
│   ├── ingestor.py                 # Regex text cleaner
│   ├── database.py                 # Star schema + ACID transactional warehouse
│   ├── ml_models.py                # TF-IDF, K-Means (Kneedle elbow), LogReg/RF baselines
│   ├── dl_lstm.py                  # BiLSTM urgency regressor (PyTorch, V1/V2)
│   ├── transformer_regressor.py    # Transfer-learned DistilBERT urgency regressor (V3)
│   ├── transformer_nlp.py          # BERT NER + BART summarization (lazy-loaded)
│   └── genai_copilot.py            # CoT/few-shot prompts + pluggable LLM backend
├── scripts/
│   ├── generate_data.py            # Generates both V1 (naive) and V2 (hardened) datasets
│   ├── run_pipeline.py             # Ingest → warehouse → train ML + BiLSTM, per variant
│   ├── train_v3.py                 # Fine-tunes the V3 DistilBERT regressor
│   ├── stress_test_compare.py      # Real-world sentences through V2 and V3, side by side
│   └── fetch_real_datasets.py      # Optional: real Yelp/Sentiment140 samples for research use
├── tests/                          # 31 pytest tests: ingestor, warehouse ACID, ML, copilot, variants
├── data/raw/, data/warehouse/       # Pre-built for both datasets — deploy needs no training step
├── models/v1_naive/, v2_hardened/, v3_transformer/  # Pre-trained artifacts for all three variants
├── Dockerfile
└── PROGRESS.md                     # Running build log / decision journal
```

---

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the pipeline from scratch

```bash
python scripts/generate_data.py                     # generates BOTH dataset variants
python scripts/run_pipeline.py v1_naive v2_hardened  # trains V1 and V2 (BiLSTM-based)
python scripts/train_v3.py                          # fine-tunes V3 (DistilBERT) on the V2 dataset
python scripts/stress_test_compare.py               # V2 vs. V3 on real-world stress-test sentences
```

Each variant gets its own `models/<variant>/training_summary.json` — the numbers behind every claim in
`PROGRESS.md`.

## Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` — data and models ship pre-committed, so it works immediately with no
setup step.

## Run tests

```bash
pytest tests/ -v
```

## Optional: enable the Claude-backed GenAI Copilot

The copilot works out of the box with a deterministic local backend (no API key needed). To use
Claude instead, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your key
(or set `ANTHROPIC_API_KEY` as an environment variable) — the backend switches automatically.

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at `app.py`.
3. **Important — set the Python version to 3.11** in the deploy dialog's "Advanced settings" (or, for
   an already-deployed app, under "Manage app" → Settings). A `runtime.txt` pinning `3.11` is included,
   but Streamlit Cloud has known issues ignoring it and defaulting to whatever the latest Python is —
   which currently has no compatible `torch` wheels in this project's pinned version range. Explicitly
   selecting 3.11 in Advanced Settings is the reliable fix.
4. (Optional) add `ANTHROPIC_API_KEY` under the app's Secrets settings.
5. The warehouse and models ship pre-committed, so once the Python version is correct the app deploys
   immediately with no training step.

---

## Evaluation Metrics

| Component | V1 Naive | V2 Hardened | V3 Transformer |
|---|---|---|---|
| K-Means silhouette (Kneedle elbow, k=9) | 1.000 (memorized, meaningless) | 0.328 (realistic) | *(shares V2's clustering)* |
| Baseline classifier macro-F1 | 1.000 | 0.914 | — |
| Urgency regressor test MSE | 0.0137 | 0.0162 | 0.0160 |
| Urgency regressor test macro-F1 | 1.000 | 0.821 | 0.825 |
| **Real-world stress-test separation** (critical − low, 8 natural-language sentences) | — | **0.022** | **0.298** |

The synthetic-test-set metrics for V2 and V3 look similar — the real difference only shows up on
natural-language text neither model was trained on verbatim, which is exactly the point. Full methodology
for both the V1→V2 and V2→V3 fixes is in [`PROGRESS.md`](./PROGRESS.md).

## Credits

Independently designed and built end-to-end: real ACID transactions with rollback tests, an elbow
method robust to templated-text artifacts, early-stopped training with an honest train/val/test split
for both the BiLSTM and the transfer-learned regressor, a pluggable zero-secrets-required GenAI backend,
a live three-way model comparison built to make both fixes demonstrable rather than just claimed, and a
31-test pytest suite.
