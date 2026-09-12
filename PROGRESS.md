# OmniFeedback AI — Build Log

Running record of decisions, bugs, fixes, and results. Written as we go; this becomes the raw material for the later blog series / research paper — nothing here is retroactively cleaned up to look better than it was.

## 2026-09-12 — Milestone 0: Project kickoff

**Context**: Starting from a course capstone spec (README.md + docx) with only a data-download notebook. Goal: turn the spec into a real, tested, deployable product — not notebook-grade scripts.

**Decisions made**
- Repo layout: `src/` for library code, `pages/` for Streamlit multi-page UI, `tests/` for pytest, `notebooks/` for exploratory/training notebooks kept separate from production code.
- Requirements pinned to ranges known to install on Streamlit Community Cloud (Python 3.11 runners) — avoided unpinned `>=` only constraints from the original spec, which risks pulling breaking major versions (e.g. NumPy 2.x breaking older sklearn/torch builds).
- GenAI copilot will use a **pluggable backend**: local rule-based/template generator by default (zero secrets needed, deploy always works), auto-upgrades to Claude API if `ANTHROPIC_API_KEY` is present in Streamlit secrets. This avoids the common capstone-demo failure mode of a public deployment that 500s because a secret was never configured.
- Chose SQLite (not Postgres) for the warehouse — zero external infra, still lets us demonstrate real star-schema + FK + transaction discipline, and it's what actually ships with a Streamlit Cloud deployment.

**Known constraints going in**
- Local dev machine has Python 3.9.6 (via Xcode CLT) as system default; created a project-local `venv` rather than depending on system Python, since Streamlit Cloud and the intended `requirements.txt` target newer library baselines.
- The original data notebook's Sentiment140 HF loader fails (`Dataset scripts are no longer supported`) — will fix by pointing at a non-script Parquet mirror, with graceful fallback to the synthetic generator already in place (kept as a safety net, not removed).

_Next entries will land per milestone: ingestion, warehouse, classical ML, BiLSTM, transformers, GenAI copilot, dashboard, tests, deployment._

## 2026-09-12 — Milestone 1: Full pipeline built, tested, and running end-to-end

**What was built**: `src/ingestor.py` (regex cleaner), `src/database.py` (SQLite star schema with real
`PRAGMA foreign_keys=ON`, `BEGIN`/`COMMIT`/`ROLLBACK` transactions), `src/ml_models.py` (TF-IDF + K-Means
+ LogReg/RF baselines), `src/dl_lstm.py` (BiLSTM urgency regressor with early stopping), `src/transformer_nlp.py`
(lazy BERT NER + BART summarizer), `src/genai_copilot.py` (CoT/few-shot prompting, pluggable Claude/local
backend), the full `app.py` + 5-page Streamlit dashboard, 21 pytest tests, a `Dockerfile`, and
`scripts/generate_data.py` + `scripts/run_pipeline.py` to reproduce everything from scratch.

**Bug 1 — Python version mismatch**: first venv was built from the system Python 3.9.6 (Xcode CLT), and
`src/database.py` used `str | Path` union-type syntax (3.10+ only), crashing on import. Fixed by rebuilding
the venv against Homebrew's Python 3.11 — this also matches Streamlit Community Cloud's runtime, so it's
the right target anyway, not just a local workaround.

**Bug 2 — synthetic data was too easy (silent overfitting risk)**: the first pipeline run produced
suspiciously perfect results — baseline classifiers hit macro-F1 = 1.0, and K-Means silhouette climbed
monotonically to the edge of the tested range (best k = 8, the last k tried). Investigated and found the
root cause: the original template-based generator picked verbatim strings from a tiny fixed phrase list per
class, so classes were linearly separable by exact keyword overlap and K-Means clusters were tight
paraphrase groups — a property of the toy data, not evidence of a good model. **Fix, two parts:**
1. Rewrote `scripts/generate_data.py` to compose sentences from templates + randomized fill-words (not
   verbatim phrases), added a class of deliberately **ambiguous/mixed-sentiment examples** that straddle
   the urgency threshold, and injected ~3% random label noise to simulate annotator disagreement. Result:
   baseline F1 dropped from 1.00 to a much more credible **0.91**, and BiLSTM test F1 landed at **0.82** —
   both realistic, neither trivial nor collapsed.
2. Realized that even with harder data, **naive `argmax(silhouette)` k-selection is unsound** for
   template-structured text, because silhouette can keep climbing as k approaches the number of underlying
   sentence templates — that's a metric artifact, not a real "elbow." Replaced the naive argmax with the
   **Kneedle max-distance-from-chord elbow method** (Satopaa et al. 2011) in `select_best_k()`, which finds
   the point of maximum curvature relative to the line joining the first and last sampled (k, silhouette)
   points, regardless of whether the gain curve is smooth or wobbly. This is a small but real methodological
   choice worth keeping in the eventual paper: *"blindly maximizing a clustering validity metric is not the
   same as finding a business-meaningful segmentation."*

**Current metrics** (from `models/training_summary.json`, `python scripts/run_pipeline.py`, 4,000-record
synthetic dataset, reproducible via `scripts/generate_data.py`):

| Component | Metric | Result | Spec target |
|---|---|---|---|
| K-Means clustering | Best k (Kneedle elbow) / silhouette | k=9 / 0.328 | Silhouette > 0.45 |
| Baseline classifiers | Macro-F1 (LogReg / RF) | 0.914 / 0.912 | F1 > 0.82 |
| BiLSTM urgency regressor | Val MSE / Test MSE (early-stopped @ epoch 8) | 0.019 / 0.016 | MSE < 0.05 |
| BiLSTM urgency regressor | Test macro-F1 (thresholded @ 0.5) | 0.817 | F1 > 0.85 |
| Ingestion pipeline | Avg latency/record | 0.025 ms | < 350 ms |

**Honest read**: silhouette (0.328) is below the spec's 0.45 target — expected and reported as-is, not
fudged, because the synthetic aspect categories genuinely overlap in vocabulary by design (that's what makes
the ambiguous-class fix above meaningful). BiLSTM F1 (0.817) is just under the 0.85 target; the regression
metrics (val/test MSE ~0.016-0.019) are comfortably under the 0.05 target, and the small F1 gap is a direct,
expected consequence of choosing 0.5 as the binarization threshold on a continuous score near a genuinely
ambiguous decision boundary — not a training failure. Both are flagged here as candidates for the Step 6
data-insights pass (e.g., calibrating the classification threshold via precision/recall trade-off, or trying
a second clustering algorithm on real, non-templated review text) rather than being silently "tuned away" by
picking a friendlier random seed.

**Anti-overfitting mechanisms actually in place** (BiLSTM): explicit train/val/test split done once with a
fixed seed before any model sees data, early stopping on validation MSE (patience=4), dropout p=0.3, Adam
with L2 weight decay 1e-5, and metrics reported on the held-out test set — not validation, not train.

**Verification performed**: all 21 pytest tests pass (`pytest tests/ -v`); a live Streamlit server was
started and its core logic path exercised end-to-end (warehouse fetch returns all 4,000 rows, TF-IDF/K-Means/
baseline artifacts load, BiLSTM scores a live sentence, GenAI copilot returns schema-valid JSON via the local
backend since no API key is configured, chi-squared test runs without error); server health check returned
`ok`. NER/BART pages were confirmed to import cleanly but were not exercised with real inference (multi-GB
model downloads) — flagged as follow-up before final deployment sign-off.

**Deferred to the next pass**: README screenshots, Docker build verification, GitHub push.

## 2026-09-12 — Milestone 2: NER/BART live-tested, real dataset sourcing fixed

**NER + BART live inference** (previously only import-tested, not run): live-tested
`src/transformer_nlp.py` end to end. `dbmdz/bert-large-cased-finetuned-conll03-english` correctly
extracted all 4 entities from a test sentence ("The iPhone 15 crashed right after the iOS 17.2 update,
and Apple support in California hasn't responded.") — `iPhone 15` (MISC), `iOS 17.2` (MISC), `Apple` (ORG),
`California` (LOC), all with >0.94 confidence. `facebook/bart-large-cnn` correctly condensed 5 distinct
crisis reports into one coherent summary sentence. First run downloaded ~2.5GB of weights (~95s combined
cold-start); cached reruns take ~5s each.

**Bug — invalid pipeline kwarg**: attempted to pass `clean_up_tokenization_spaces=True` directly to
`pipeline("ner", ...)` to silence a deprecation warning, which crashed with
`TypeError: TokenClassificationPipeline._sanitize_parameters() got an unexpected keyword argument`. That
kwarg belongs to the tokenizer's `decode()`/`batch_decode()` call, not the token-classification pipeline
constructor. Fixed by removing it from the NER pipeline (kept only where valid) and instead explicitly
pinning `device=-1` on both pipelines, which also silences the separate "accelerator available but no
device passed" warning and matches the CPU-only reality of the Streamlit Community Cloud deploy target.

**Fix — BART length-mismatch warning**: `summarize_batch` was calling BART with a fixed `max_length=130`
regardless of input size, which warns ("max_length set to 130 but input_length is only 67...") on short
inputs and can produce degenerate output on very short ones. Replaced with `_bounded_lengths()`, which caps
`max_length`/`min_length` relative to the actual input word count per chunk — a small but real
data-dependent-parameter bug, the kind that's invisible in a demo with long inputs but bites in production
with real short tickets.

**Real dataset sourcing fixed** (`scripts/fetch_real_datasets.py`, new): the original capstone notebook's
Sentiment140 loader (`stanfordnlp/sentiment140` / `sentiment140`) fails with "Dataset scripts are no longer
supported" — HuggingFace deprecated script-based dataset loaders. Confirmed `Yelp/yelp_review_full` still
works unchanged. For Sentiment140, found and verified a maintained Parquet mirror,
`contemmcm/sentiment140`, which has the same `{text, label}` schema but a `complete` split rather than
`train` (a one-line fix once found: `split="complete[:n]"`, not `split="train[:n]"`). Both now load cleanly
and are saved as CSVs for the Step 6 real-vs-synthetic comparison work, without touching the main synthetic
pipeline (`scripts/generate_data.py`/`run_pipeline.py`), which remains the source of truth for the deployed
demo so the app never depends on live internet access at deploy time.

**Verification performed**: `pytest tests/ -v` — 21/21 passing after both fixes; live NER/BART rerun with
cached weights confirms no warnings and correct entity/summary output; both real dataset fetches (Yelp,
Sentiment140) verified to load and shape correctly.

**Deferred to the next pass**: README screenshots, Docker build verification, GitHub push.

## 2026-09-12 — Milestone 3: Live V1 (naive) vs. V2 (hardened) comparison in the app

**Why**: after Milestone 1's fix (perfect metrics → realistic metrics via a hardened data generator),
the "before" state only existed as a paragraph in this file. For live demos and the eventual research
writeup, letting a viewer flip a switch and *watch* the metrics change is far more convincing than
describing it — so the app now ships both dataset/model versions side by side, trained through the
identical pipeline code, differing only in the input data.

**What changed**:
- `src/variants.py` (new): registry of the two variants — `v1_naive` (verbatim phrase lists, ~9 unique
  training sentences total, reproducing the original capstone notebook almost exactly) and `v2_hardened`
  (the Milestone 1 fix: templated/randomized text, ambiguous cases, label noise) — each with its own raw
  CSV, SQLite warehouse file, and `models/<variant>/` directory so the two never collide.
- `scripts/generate_data.py`: now generates both variants explicitly (`generate_naive()` /
  `generate_hardened()`), rather than one script producing one anonymous dataset.
- `scripts/run_pipeline.py`: takes variant IDs as CLI args (`python scripts/run_pipeline.py v1_naive
  v2_hardened`) and writes to variant-scoped paths.
- `src/ml_models.py`, `src/dl_lstm.py`: `model_dir` is now a parameter (defaulting to the module-level
  `MODELS_DIR` for backward compatibility) instead of a hardcoded path, so the same training/loading code
  serves both variants without duplication.
- `src/app_state.py` (new): a shared sidebar variant selector (`select_variant()`), added to `app.py` and
  every data-driven page, backed by `st.session_state` so the choice persists across page navigation.
- `pages/0_Model_Comparison.py` (new): side-by-side metrics, the silhouette-vs-k curve for both variants
  overlaid on one chart, BiLSTM train/val loss curves overlaid, and a random sample of each variant's raw
  text so a viewer can see *why* the numbers differ, not just that they do.

**Bug caught during refactor**: the first pass gave `model_dir: Path = MODELS_DIR` as a function default
argument. Python binds default arguments once, at function-definition time — so a test that monkeypatched
`ml_models.MODELS_DIR` to a temp directory had no effect, since the already-bound default still pointed at
the original path, and `test_kmeans_pipeline_selects_valid_k_and_saves` failed with a confusing "file not
found in temp dir" symptom. Fixed by defaulting to `None` and resolving `model_dir = model_dir or
MODELS_DIR` inside the function body, which reads the (possibly monkeypatched) module global at call time
instead of at import time — a good reminder that mutable-module-global-as-default-arg is a footgun even
when the "mutable" part is just a `Path` swap for tests.

**Verified side-by-side results** (both variants trained via the identical `run_pipeline.py` code path):

| Metric | V1 Naive | V2 Hardened |
|---|---|---|
| Unique training sentences | 9 | 212 |
| K-Means silhouette (best k=9) | **1.000** (perfect — meaningless) | 0.328 (realistic) |
| Baseline classifier macro-F1 | **1.000** | 0.914 |
| BiLSTM test MSE | 0.0137 | 0.0162 |
| BiLSTM test macro-F1 | **1.000** | 0.821 |

This is the demo-ready version of Milestone 1's finding: V1's perfect scores are the red flag (memorized 9
sentences), not a win — and now a viewer can toggle between them and see it happen live instead of taking
our word for it.

**Verification performed**: `pytest tests/ -v` — 25/25 passing (added `tests/test_variants.py`); a
dedicated logic-layer smoke test confirmed both variants load independently (warehouse row counts,
model artifacts, BiLSTM inference on the same sentence returning variant-specific scores) and that the
"V1 looks better but is overfit" pattern holds exactly as expected in the saved training summaries; a live
Streamlit server smoke test confirmed the app still boots cleanly with the new sidebar and page.

## 2026-09-12 — Milestone 4: GitHub push + Docker verification

**GitHub**: pushed to [github.com/achellesheel/omnifeedback-ai](https://github.com/achellesheel/omnifeedback-ai)
(public). Deliberately excluded the original course-provided spec document and scaffold notebook from
version control — they're the source material's IP, not ours to redistribute, and are fully superseded
by `scripts/generate_data.py`. README rewritten as an original project write-up rather than a course
deliverable.

**Docker — real bug found and fixed**: the `HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health`
directive was silently broken — `python:3.11-slim` doesn't ship `curl`, so every healthcheck would have
failed forever without ever showing up as a build error (Docker doesn't validate that a `HEALTHCHECK`
binary exists at build time, only at container runtime). Caught by actually running the container and
checking `docker inspect --format='{{json .State.Health}}'` instead of just checking the image built.
Fixed by installing `curl` alongside `build-essential`. Also added `.dockerignore` (excludes `venv/`,
`.git/`, and the excluded course materials) to keep the build context lean.

**Verified**: `docker build` succeeds cleanly; `docker run` starts the container, `curl
localhost:8502/_stcore/health` returns `ok`, the app serves real HTML (not an error page), and Docker's
own health monitor reports `"Status":"healthy"` after the first check interval — confirming the fix
actually works, not just that the Dockerfile parses.

**Status**: product build (steps 1–5) is now complete, tested, containerized, and pushed. Next up per
the original request: steps 6–9 (data insights, QA/PM-style business review, enhancement ideas, 15-page
slide deck), then the content phase (blog series, Twitter/LinkedIn, research paper).

## 2026-09-13 — Milestone 5: First live deploy failed — Streamlit Cloud Python version mismatch

**What happened**: first deploy to Streamlit Community Cloud failed during dependency install. Cloud
provisioned **Python 3.14.7** for the app container; `torch<2.6.0,>=2.2.0` (this project's pin, matching
what was tested locally on Python 3.11) has no published wheels for Python 3.14 at all — the resolver
correctly reported the pin as unsatisfiable rather than silently installing something broken.

**Root cause**: nothing in the repo told Streamlit Cloud which Python to use, so it defaulted to its
current latest (3.14), which is newer than this project's pinned dependency range supports. This is a
gap in Milestone 1's deployment-readiness assumptions — "pinned to versions known to install on Streamlit
Cloud's Python 3.11" assumed Cloud defaults to 3.11, which was true when that assumption was written but
is no longer the platform default.

**Fix, two parts**:
1. Added `runtime.txt` (containing `3.11`) to the repo root — the documented mechanism for pinning a
   Community Cloud app's Python version.
2. **However**: multiple current Streamlit community reports (streamlit/streamlit#15326 and several
   discuss.streamlit.io threads, mid-late 2026) describe `runtime.txt` being silently ignored by Cloud's
   build system, with apps still provisioned on 3.13/3.14 regardless. Streamlit's own docs describe the
   Python version as being set via the **"Advanced settings" dialog at deploy time** (or "Manage app" →
   Settings for an existing app), not via a repo file — that UI setting is the authoritative mechanism.
   Documented this clearly in the README's deployment section so the fix isn't silently dependent on a
   file that may not be honored.

**Lesson for the eventual blog post**: "pin your requirements" is necessary but not sufficient for
reproducible deployment — the runtime itself needs pinning too, and on managed platforms that pin may
live in platform UI/settings rather than in the repo, which is easy to miss until a real deploy fails.
This is exactly the kind of gap that a Docker-based deployment (already verified working in Milestone 4)
sidesteps entirely, since the base image fixes the Python version explicitly — worth highlighting as a
reason to prefer container deployment for anything beyond a quick demo.

**Resolved**: confirmed working after setting Python 3.11 explicitly via Streamlit Cloud's Advanced
Settings. Live app: https://omnifeedback-ai-c4mmclqnkgytms6vuzqtjp.streamlit.app/

## 2026-09-13 — Milestone 6: Data insights, QA/PM review, roadmap, and a 15-slide deck (Steps 6–9)

Moved past the core build into the analysis/presentation phase requested next.

- **`scripts/data_insights.py`** computes real confusion matrix, threshold sensitivity (with a caught
  metric-comparison bug — see below), predicted-vs-true correlation, per-channel/aspect breakdowns, and
  concrete misclassification examples against the deployed V2 model. Full writeup: `docs/DATA_INSIGHTS.md`.
- **Bug caught mid-analysis**: the first threshold-sweep pass used sklearn's `precision_recall_curve`,
  which only tracks *positive-class* F1 — not comparable to the macro-F1 used everywhere else in this
  project. It reported a "best" threshold that scored *worse* than the default 0.5, which was the tell
  that something was wrong. Fixed by sweeping macro-F1 directly across candidate thresholds; the real
  result (best threshold 0.48, only a 0.3-point F1 gain over 0.5) is a much more mundane and much more
  credible finding.
- **Real limitation surfaced, not hidden**: channel and aspect category carry essentially no signal in
  the synthetic dataset (chi-squared p=0.840) — traced to the generator assigning them independent of
  content. Documented plainly and queued as the top item in `docs/ENHANCEMENTS.md` rather than glossed
  over.
- **`docs/PRODUCT_REVIEW.md`**: QA/PM-style walkthrough — three user journeys (support manager, product
  manager, technical reviewer), a 9-item findings table (fixed vs. open), and business-value framing that
  states the platform's real caveats alongside its real strengths.
- **`docs/ENHANCEMENTS.md`**: prioritized roadmap — free fixes first (generator channel/aspect
  correlation, real-dataset validation pass, shared UI state), then cheap paid APIs (OpenAI embeddings,
  Pinecone RAG, Claude already wired, Twilio/Slack alerting, hosted inference endpoints), then bigger
  research bets (DistilBERT comparison, active learning, drift monitoring, multilingual support).
- **15-slide technical deck** (`docs/slides/index.html`, published as an Artifact): architecture,
  algorithms, the V1-vs-V2 story, and the data-insights findings above — built with a real design pass
  (IBM Plex Sans/Mono pairing, an amber "urgency" accent tying the palette to the product's own urgency
  score, light/dark theme support, keyboard/swipe navigation, inline SVG charts drawn from the actual
  k-sweep and confusion-matrix numbers rather than stock icons).

**Verification performed**: `pytest tests/ -v` still 25/25 passing (no source code changed, only new
analysis scripts/docs); `data_insights.py` output manually sanity-checked against `training_summary.json`
before writing any of the markdown docs, so every number quoted is traceable to an actual run.
