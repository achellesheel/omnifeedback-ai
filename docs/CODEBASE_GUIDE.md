# Codebase Guide

A file-by-file map of OmniFeedback AI: what each file does, every class/function in it in plain
English, and a glossary of the ML/DL/NLP terms used throughout. Written as a companion to
[`PROGRESS.md`](../PROGRESS.md) (the *why*/build-log) — this doc is the *what's where*.

---

## 1. How data actually moves through the system

```
data/raw/*.csv                          (synthetic, or real Yelp/Sentiment140 samples)
     │
     │  scripts/generate_data.py  →  writes the raw CSVs in the first place
     ▼
scripts/run_pipeline.py
     │  TextIngestor.clean_text()  — in-memory pandas column (src/ingestor.py)
     ▼
SQLite warehouse: data/warehouse/*.db    (src/database.py, WarehouseManager)
     │  Fact_Feedback table stores BOTH raw_text and cleaned_text
     │
     ├──► run_ml_pipeline()            (src/ml_models.py)     → models/<variant>/*.pkl
     ├──► train_bilstm_model()         (src/dl_lstm.py)        → models/<variant>/bilstm_urgency.pth
     └──► train_transformer_regressor()(src/transformer_regressor.py, scripts/train_v3.py)
                                                                → models/v3_transformer/transformer_urgency.pth

Live Streamlit app (app.py + pages/0-5)
     │  reads back via WarehouseManager.fetch_all_feedback()
     │  scores new/live text via src/urgency.py → src/dl_lstm.py or src/transformer_regressor.py
     │  NER/summarization via src/transformer_nlp.py (+ optional src/cloud_ner.py)
     └─ resolution drafting via src/genai_copilot.py
```

Three "variants" (V1/V2/V3) run this same pipeline with different data-generation strategies and
different urgency models, registered in `src/variants.py` and toggleable live in every page.

---

## 2. `src/` — the reusable core

### `src/ingestor.py` — text cleaning
| Name | What it does |
|---|---|
| `TextIngestor` (class) | Namespace for the regex-based cleaning methods below — no state, all `@classmethod`. |
| `clean_text(text)` | Lowercases, strips HTML tags/URLs/@mentions, keeps `#hashtag` words (drops the `#`), collapses repeated letters ("sooooo"→"soo"), strips remaining punctuation, collapses whitespace. This is what feeds the statistical models (TF-IDF, BiLSTM). |
| `extract_hashtags(text)` / `extract_mentions(text)` | Pull out `#tags` / `@mentions` as metadata, separately from the cleaned body text. |

### `src/database.py` — the SQL warehouse
| Name | What it does |
|---|---|
| `SCHEMA_SQL` | The star-schema DDL: `Fact_Feedback` (facts) + `Dim_User`/`Dim_Channel`/`Dim_Product` (dimensions), with foreign keys and a `CHECK` constraint on `urgency_score` (must be 0–1). |
| `WarehouseManager` (class) | Wraps one SQLite database file. |
| `get_connection()` | Context-manager: opens a connection with `PRAGMA foreign_keys = ON`, always closes it. |
| `_init_schema()` | Runs `SCHEMA_SQL` once at construction (idempotent — `CREATE TABLE IF NOT EXISTS`). |
| `_get_or_create_dim(...)` | Look-up-or-insert helper for the two dimension tables (e.g. gets `channel_id` for `"Twitter"`, inserting the row if it's new). |
| `insert_feedback_batch(df)` | Inserts a whole DataFrame inside **one transaction** (`BEGIN`/`COMMIT`) — if any row fails a constraint, the *entire batch* rolls back, so partial/corrupt writes never land. |
| `query(sql, params)` | Generic escape hatch — any SQL, returned as a DataFrame. |
| `fetch_all_feedback()` | The one query every page actually uses: joins `Fact_Feedback` back with its two dimension tables to return a flat, human-readable DataFrame. |

### `src/ml_models.py` — classical ML (clustering + baseline classifiers)
| Name | What it does |
|---|---|
| `fit_tfidf(texts)` | Fits a **TF-IDF** vectorizer (see glossary) on the cleaned text. |
| `select_k_via_silhouette(X)` | Runs K-Means for a range of `k` values, records the **silhouette score** for each. |
| `select_best_k(k_scores)` | Picks the best `k` using the **Kneedle** elbow method (max distance from the chord connecting the first/last points) instead of naively taking the highest silhouette score — the naive approach turned out to always pick the largest `k` tested on templated text (see `PROGRESS.md`). |
| `fit_kmeans(X, n_clusters)` | Fits the final K-Means model at the chosen `k`. |
| `train_baseline_classifiers(...)` | Trains Logistic Regression + Random Forest classifiers on the urgency label, purely as a non-deep-learning comparison point. |
| `run_ml_pipeline(df)` | Orchestrates all of the above for one variant and saves the fitted TF-IDF/K-Means/classifier artifacts to disk. |
| `load_artifacts(model_dir)` | Loads those saved artifacts back for the live app (`pages/2_Clustering_Explorer.py`). |

### `src/dl_lstm.py` — the from-scratch deep learning model (V1/V2)
| Name | What it does |
|---|---|
| `TextVocabulary` (class) | Builds a word→integer vocabulary **from scratch** from the training text (this is the "closed vocabulary" that turned out to be the root cause of V2's real-world failures — only 196 words). `encode()` turns a sentence into a fixed-length integer sequence; `to_dict()`/`from_dict()` serialize it alongside the model. |
| `FeedbackDataset` (class) | A PyTorch `Dataset` wrapping (text, urgency-score) pairs for the `DataLoader`. |
| `BiLSTMUrgencyRegressor` (class) | The model itself: embedding layer → **BiLSTM** → dropout → sigmoid output (0.0–1.0 urgency score). |
| `train_bilstm_model(df)` | Full training loop: train/val/test split (fixed seed, done once), early stopping on validation loss, dropout + L2 weight decay as anti-overfitting measures. |
| `load_bilstm_model(model_dir)` | Reloads a trained model + its vocabulary for inference. |
| `predict_urgency(model, vocab, text)` | Scores one piece of text. |

### `src/transformer_regressor.py` — the transfer-learned model (V3)
| Name | What it does |
|---|---|
| `UrgencyTextDataset` (class) | Same job as `FeedbackDataset` above, but tokenizes with the DistilBERT subword tokenizer instead of the from-scratch vocabulary — this is what removes the closed-vocabulary/OOV problem entirely. |
| `TransformerUrgencyRegressor` (class) | Wraps a pretrained `distilbert-base-uncased`, **freezes** the embeddings + first 4 of 6 transformer layers, adds a small trainable regression head on top. |
| `trainable_param_fraction()` | Reports what fraction of parameters actually update during training (~21%). |
| `trainable_state_dict()` | Returns *only* the fine-tuned parameters (not the frozen, unmodified backbone) — this is what shrank the saved checkpoint from 265MB to 57MB, small enough for a normal GitHub push. |
| `train_transformer_regressor(...)` | The fine-tuning loop (only the unfrozen layers + head receive gradient updates). |
| `load_transformer_regressor(model_dir)` | Reconstructs the frozen backbone fresh from HuggingFace, then loads the slim fine-tuned checkpoint on top. |
| `predict_urgency_transformer(...)` | Scores one piece of text through the transformer path. |

### `src/urgency.py` — routes to whichever model a variant uses
| Name | What it does |
|---|---|
| `load_urgency_model(variant)` | Looks at `variant["model_type"]` and loads either the BiLSTM bundle or the transformer bundle — callers don't need to know which. |
| `score_urgency(bundle, text)` | Scores text through whichever model was loaded — the single call site every page uses, regardless of which variant is active. |

### `src/transformer_nlp.py` — entity extraction & summarization (shared by all variants)
| Name | What it does |
|---|---|
| `get_ner_pipeline()` | Lazily loads a HuggingFace **NER** pipeline (`dslim/bert-base-NER`), cached so it only loads once per session. |
| `get_summarizer_pipeline()` | Lazily loads a HuggingFace summarization pipeline (`sshleifer/distilbart-cnn-12-6`, a **BART**-family model). |
| `extract_entities(text)` | Runs NER, returns a list of `{entity_group, word, score}` dicts. |
| `summarize_batch(texts)` | Concatenates a batch of tickets, chunks them to fit the model's token limit, summarizes each chunk, and — if there was more than one chunk — summarizes the summaries once more. |
| `_bounded_lengths(text)` | Internal helper: keeps `max_length`/`min_length` sane relative to how long the actual input is, so the model doesn't warn about asking for a longer summary than the input itself. |

### `src/cloud_ner.py` — optional V4 entity extraction (Google Cloud NL API)
| Name | What it does |
|---|---|
| `_get_gcp_api_key()` | Reads `GOOGLE_CLOUD_API_KEY` from environment or Streamlit secrets. |
| `cloud_nl_available()` | Whether a key is configured. |
| `extract_entities_cloud_nl(text)` | Calls Google's Cloud Natural Language API instead of the local BERT model — its entity schema includes a `CONSUMER_GOOD` type, which is exactly the tag CoNLL-2003 (PER/ORG/LOC/MISC only) lacks, so product names like "iPhone 15" get tagged correctly instead of falling into `MISC`. Purely additive — doesn't touch `transformer_nlp.py`. |

### `src/genai_copilot.py` — the GenAI resolution copilot
| Name | What it does |
|---|---|
| `_risk_level_from_score(score)` | Maps a 0–1 urgency score to CRITICAL/HIGH/MEDIUM/LOW. |
| `_local_backend(text, score)` | Deterministic, zero-cost fallback: keyword-pattern matching + templated JSON output — works with no API key at all. |
| `_get_anthropic_key()` | Reads `ANTHROPIC_API_KEY` from environment or Streamlit secrets. |
| `_claude_backend(text, score, api_key)` | Calls Claude with a **Chain-of-Thought** system prompt + few-shot examples to draft a resolution. |
| `_parse_json_response(raw)` | Strips markdown fences and extracts/validates the JSON object from an LLM's raw text response. |
| `generate_resolution(text, score)` | Public entry point: tries Claude if a key is configured, otherwise the local backend — always returns the same JSON schema either way. |
| `_claude_executive_summary(texts, api_key)` | Sends a batch of tickets to Claude with an explicit "synthesize themes, don't copy sentences" system prompt. |
| `generate_executive_summary(texts)` | Public entry point: Claude if configured, otherwise falls back to `summarize_batch()` from `transformer_nlp.py` (labeled honestly as more extractive on this fallback path). |

### `src/variants.py`, `src/app_state.py`, `src/ui_colors.py` — plumbing
| Name | What it does |
|---|---|
| `VARIANTS` (dict) | The registry of V1/V2/V3: each entry's raw CSV path, database path, model directory, and `model_type` ("bilstm" or "transformer"). No functions — just configuration. |
| `select_variant()` | Renders the sidebar radio button used on every page, persists the choice in `st.session_state`. |
| `RISK_LEVEL_COLORS` / `STREAMLIT_VALID_COLORS` | Constants mapping risk levels to Streamlit's actually-supported color names (a real bug: Streamlit has no "yellow"). |

---

## 3. `scripts/` — offline, one-off scripts (never run by the live app)

| File | What it does |
|---|---|
| `generate_data.py` | `generate_naive()` (V1: ~9 fixed phrases) and `generate_hardened()` (V2: templated + randomized + ambiguous + label noise) — produces the raw training CSVs. |
| `run_pipeline.py` | `run(variant_id)` — the orchestrator described in §1: clean → load into warehouse → train classical ML → train BiLSTM. |
| `train_v3.py` | Fine-tunes the V3 DistilBERT regressor on the same V2 data. |
| `fetch_real_datasets.py` | `fetch_yelp()` / `fetch_sentiment140()` — pulls real, independently-authored review/tweet samples for external validation. |
| `validate_real_data.py` | Re-runs the V2-vs-V3 separation test on the real Yelp/Sentiment140 samples (statistical tests below), instead of only on hand-written examples. |
| `stress_test_compare.py` | The original 8 hand-written sentences used to first expose V2's failure and confirm V3's fix. |
| `superiority_scorecard.py` | Consolidates every V2-vs-V3 comparison metric into one JSON the app displays. |
| `data_insights.py` | General exploratory statistics on the generated dataset (label distribution, channel/aspect crosstabs, etc.) written up in `DATA_INSIGHTS.md`. |

---

## 4. `pages/` and `app.py` — the live Streamlit app

| File | What it shows | Key calls |
|---|---|---|
| `app.py` | Executive Overview: KPI tiles, urgency-by-channel boxplot, volume-by-aspect bar chart, urgency trend over time, recent critical tickets table. | `select_variant()`, `WarehouseManager.fetch_all_feedback()` |
| `pages/0_Model_Comparison.py` | V1→V2→V3 metrics side by side, the Kneedle elbow chart, the V2-vs-V3 superiority scorecard, real-data (Yelp/Sentiment140) validation charts. | reads `training_summary.json` + the JSON files from `scripts/superiority_scorecard.py` / `validate_real_data.py` |
| `pages/1_Inference_Playground.py` | Paste any text, get a live urgency score. | `TextIngestor.clean_text()` (for the BiLSTM path) → `load_urgency_model()` / `score_urgency()` |
| `pages/2_Clustering_Explorer.py` | K-Means cluster contents, the elbow/silhouette chart that justifies the chosen `k`. | `load_artifacts()`; falls back to V2's clustering artifacts when V3 is selected, since V3 only replaces the urgency regressor |
| `pages/3_GenAI_Copilot.py` | Paste text → urgency score → drafted resolution (root cause, risk level, recommended action, customer reply). | `score_urgency()` → `generate_resolution()` |
| `pages/4_NER_and_Summarization.py` | Batch executive summary (deduplicated, top-N by urgency) + a ticket-viewer expander + entity extraction on a sample sentence. | `generate_executive_summary()`, `extract_entities()` |
| `pages/5_Statistical_Tests.py` | Chi-squared test of independence between feedback channel and urgency bucket. | `scipy.stats.chi2_contingency` over a bucketed contingency table |

---

## 5. Glossary — every technical term used above, in plain English

**Model architectures**
- **BiLSTM** — *Bidirectional Long Short-Term Memory* network. An **LSTM** (a type of recurrent neural network built to remember information over long sequences) that reads a sentence both forward and backward, then combines both readings — used here as the from-scratch V1/V2 urgency model.
- **DistilBERT** — a smaller, distilled ("compressed") version of **BERT** (*Bidirectional Encoder Representations from Transformers*), a transformer model pretrained on huge amounts of general text to already "understand" language before it's ever shown our data. V3 fine-tunes this instead of learning language from scratch.
- **BART** — *Bidirectional and Auto-Regressive Transformer*, a transformer pretrained to reconstruct corrupted text; the model family used here for summarization (specifically `distilbart-cnn-12-6`, a distilled version fine-tuned on the CNN/DailyMail news-summarization dataset).
- **Transformer** — the neural network architecture (introduced in "Attention Is All You Need") underlying BERT, DistilBERT, BART, and GPT/Claude-style models; processes a whole sequence at once using an "attention" mechanism instead of reading it step by step like an LSTM does.
- **NER** — *Named Entity Recognition*: tagging spans of text as a person, organization, location, etc.
- **CoNLL-2003** — the dataset/tagging scheme (PER/ORG/LOC/MISC — only 4 categories) most small NER models, including ours, are trained on. No dedicated "product" tag exists in it, which is why "iPhone 15" gets tagged `MISC` (see `PROGRESS.md` Milestone 16).
- **OOV** — *Out-Of-Vocabulary*: a word the model has never seen and therefore can't represent — the root cause of the from-scratch BiLSTM's real-world failures.

**Classical ML**
- **TF-IDF** — *Term Frequency–Inverse Document Frequency*: a way to turn text into numbers by weighting words that are frequent in one document but rare across all documents (so common filler words score low, distinctive words score high).
- **K-Means** — an unsupervised clustering algorithm that groups data points into `k` clusters by iteratively assigning each point to its nearest cluster center.
- **Silhouette score** — a metric (−1 to 1) for how well-separated clusters are; higher means points fit their own cluster better than neighboring ones.
- **Kneedle** — an algorithm for finding the "elbow" (point of maximum curvature) in a curve, used here to pick `k` honestly instead of just taking whichever `k` scores highest.
- **F1 score** — the harmonic mean of precision and recall; a single number summarizing classifier accuracy.
- **MSE** — *Mean Squared Error*: the average squared difference between predicted and actual values, used to evaluate the urgency regressor.

**Statistics**
- **Chi-squared test (χ²)** — a statistical test for whether two categorical variables (here: channel and urgency bucket) are independent of each other.
- **Mann-Whitney U test** — a non-parametric test for whether two groups' distributions differ, used to compare V2 vs. V3 scores on real data without assuming a normal distribution.
- **Spearman correlation (ρ)** — measures how well the relationship between two variables can be described by a monotonic (consistently increasing/decreasing) function, used to compare model scores against Yelp's 5-star ratings.
- **p-value** — the probability of seeing a result this extreme if there were actually no real effect; conventionally, p < 0.05 is called "statistically significant."

**GenAI / LLM**
- **GenAI** — *Generative AI*: models (like Claude) that generate new text/content rather than just classifying or scoring it.
- **Chain-of-Thought (CoT)** — a prompting technique that asks a model to reason step by step before giving its final answer, generally improving answer quality.
- **Few-shot prompting** — giving a model a few worked examples in the prompt itself, so it infers the expected format/style without any retraining.
- **Pluggable backend** — a design pattern used in `genai_copilot.py`: the same function can be served by different implementations (Claude API vs. a free local rule-based fallback) depending on what's configured, with identical output shape either way.

**Systems / data**
- **ACID transaction** — a database write that is Atomic, Consistent, Isolated, and Durable — in practice here, it means a batch insert either fully commits or fully rolls back, never leaves the database half-written.
- **Star schema** — a database design with one central "fact" table (here, `Fact_Feedback`) referencing several smaller "dimension" tables (`Dim_User`, `Dim_Channel`, `Dim_Product`) via foreign keys, instead of repeating that descriptive data on every row.
- **API key** — a secret credential used to authenticate calls to an external service (Anthropic's Claude, Google Cloud); never something to commit to version control (see `.streamlit/secrets.toml`, which is git-ignored).
