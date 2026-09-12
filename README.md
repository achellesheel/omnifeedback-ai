# OmniFeedback AI

**An enterprise customer-feedback intelligence platform: ingestion, escalation triage, and a GenAI resolution copilot — with the overfitting fix made visible, live.**

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.3x-red.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/tests-25%20passing-green.svg)](./tests)

Most feedback-analytics demos show you one model and ask you to trust the metric. This one shows you
**two** — the same pipeline trained on naive data versus hardened data — so you can watch a "perfect"
macro-F1 of 1.0 turn out to be memorization, not intelligence, and see what an honestly-evaluated
model looks like instead.

Under the hood: multi-channel feedback ingestion through a regex cleaner into a transactional SQL star
schema, unsupervised K-Means aspect clustering (with a proper Kneedle-elbow k-selection, not a naive
argmax), a custom PyTorch BiLSTM continuous urgency regressor with early stopping, HuggingFace BERT NER
+ BART executive summarization, and a Chain-of-Thought GenAI resolution copilot with a pluggable backend
that needs zero API keys to run.

📓 See [`PROGRESS.md`](./PROGRESS.md) for the full build log — every bug hit, every fix, and the
reasoning behind each modeling choice, written as it happened.

📊 [15-slide technical walkthrough](./docs/slides/index.html) · 📈 [Data insights](./docs/DATA_INSIGHTS.md)
· 🔍 [Product review](./docs/PRODUCT_REVIEW.md) · 🚀 [Enhancement roadmap](./docs/ENHANCEMENTS.md)

---

## Live Demo Feature: V1 vs. V2

Open the **Model Comparison** page and flip between:
- **V1 Naive** — verbatim phrase lists, ~9 unique training sentences. Silhouette = 1.0, macro-F1 = 1.0.
  Looks perfect. Isn't — the model memorized 9 sentences.
- **V2 Hardened** — templated/randomized text, ambiguous mixed-sentiment cases, simulated label noise.
  Silhouette = 0.33, macro-F1 = 0.91/0.82. Lower numbers, real generalization.

Every other page (Inference Playground, Clustering Explorer, GenAI Copilot, Statistical Tests) has the
same sidebar toggle, so you can watch clusters, urgency scores, and chi-squared results shift with the
underlying data in real time.

---

## Architecture

```
Raw feedback (CSV) ──▶ Regex TextIngestor ──▶ SQLite Star Schema Warehouse (ACID transactions)
                                                              │
        ┌─────────────────────────────┬────────────────────────────┬──────────────────────┐
        ▼                             ▼                            ▼                      │
  TF-IDF + KMeans              PyTorch BiLSTM                BERT NER +               (all read from
  aspect clustering            urgency regressor             BART summarizer           the warehouse)
  (Kneedle elbow)              (early stopping,                                          │
  + LogReg/RF baseline          dropout, L2 decay)                                        │
        └─────────────────────────────┴────────────────────────────┴──────────────────────┘
                                              │
                                    GenAI Resolution Copilot
                                (CoT + few-shot, pluggable backend:
                                 local rules ↔ Claude API)
                                              │
                                  Streamlit Multi-Page Dashboard
                                    (V1 / V2 toggle on every page)
```

---

## Repository Structure

```
├── app.py                        # Streamlit entrypoint: Executive Overview
├── pages/
│   ├── 0_Model_Comparison.py      # V1 vs. V2 side by side — the headline feature
│   ├── 1_Inference_Playground.py  # Live BiLSTM urgency scoring
│   ├── 2_Clustering_Explorer.py   # K-Means clusters + elbow/silhouette justification
│   ├── 3_GenAI_Copilot.py         # CoT/few-shot resolution drafting
│   ├── 4_NER_and_Summarization.py # BERT NER + BART executive briefing
│   └── 5_Statistical_Tests.py     # Chi-squared channel/urgency independence test
├── src/
│   ├── variants.py                # V1/V2 registry (paths, labels, descriptions)
│   ├── app_state.py                # Shared sidebar variant selector
│   ├── ingestor.py                 # Regex text cleaner
│   ├── database.py                 # Star schema + ACID transactional warehouse
│   ├── ml_models.py                # TF-IDF, K-Means (Kneedle elbow), LogReg/RF baselines
│   ├── dl_lstm.py                  # BiLSTM urgency regressor (PyTorch)
│   ├── transformer_nlp.py          # BERT NER + BART summarization (lazy-loaded)
│   └── genai_copilot.py            # CoT/few-shot prompts + pluggable LLM backend
├── scripts/
│   ├── generate_data.py            # Generates both V1 (naive) and V2 (hardened) datasets
│   ├── run_pipeline.py             # Ingest → warehouse → train ML + BiLSTM, per variant
│   └── fetch_real_datasets.py      # Optional: real Yelp/Sentiment140 samples for research use
├── tests/                          # 25 pytest tests: ingestor, warehouse ACID, ML, copilot, variants
├── data/raw/, data/warehouse/       # Pre-built for both variants — deploy needs no training step
├── models/v1_naive/, models/v2_hardened/  # Pre-trained artifacts for both variants
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
python scripts/run_pipeline.py v1_naive v2_hardened  # trains BOTH variants (or pass just one)
```

Each variant gets its own warehouse DB and `models/<variant>/training_summary.json` — the numbers
behind every claim in `PROGRESS.md`.

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

| Component | V1 Naive | V2 Hardened |
|---|---|---|
| K-Means silhouette (Kneedle elbow, k=9) | 1.000 (memorized, meaningless) | 0.328 (realistic) |
| Baseline classifier macro-F1 | 1.000 | 0.914 |
| BiLSTM test MSE | 0.0137 | 0.0162 |
| BiLSTM test macro-F1 | 1.000 | 0.821 |
| Ingestion latency | 0.025 ms/record | 0.025 ms/record |

Full methodology, the Kneedle elbow-selection fix, and honest gap-analysis against target metrics are
in [`PROGRESS.md`](./PROGRESS.md).

## Credits

Independently designed and built end-to-end: real ACID transactions with rollback tests, an elbow
method robust to templated-text artifacts, early-stopped BiLSTM training with an honest train/val/test
split, a pluggable zero-secrets-required GenAI backend, a live naive-vs-hardened comparison built to make
the overfitting fix demonstrable rather than just claimed, and a 25-test pytest suite.
