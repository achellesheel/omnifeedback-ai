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

## 2026-09-13 — Milestone 7: "Risk Level: :yellow" bug — found by a real user, not by testing

**Found by**: a friend testing the live app for feedback, not by any automated check. Every MEDIUM-risk
result on the Inference Playground and GenAI Copilot pages rendered as the literal broken text
`Risk Level: :yellow` instead of colored "MEDIUM" text — for every single test case they tried.

**Root cause**: Streamlit's `:color[text]` markdown color-annotation syntax only recognizes a fixed
keyword set — confirmed directly from Streamlit's own source docstring:
`supported colors: blue, green, orange, red, violet, gray/grey, rainbow`. `"yellow"` was never a valid
keyword. When Streamlit hits an unrecognized color name inside `:color[...]`, it doesn't raise an error
or fall back to plain text — it fails to parse the whole construct, which is why the bracketed level text
disappeared entirely rather than showing as plain unstyled text.

**Why pytest didn't catch it**: this class of bug — a markdown string that parses fine as a Python
f-string but fails Streamlit's own internal color-keyword validation — is invisible to unit tests that
only check Python-level correctness. It required actually looking at the rendered page, which is exactly
why "show it to a real person" surfaced it and 27 passing tests didn't.

**Fix**: extracted the risk-level → color mapping (previously duplicated independently in both
`pages/1_Inference_Playground.py` and `pages/3_GenAI_Copilot.py` — itself part of the problem, since a
fix to one copy wouldn't have caught the other) into a single `src/ui_colors.py` constant, changed
`MEDIUM`'s color from `"yellow"` to `"violet"`, and added `tests/test_ui_colors.py` asserting every
mapped color is in Streamlit's actual supported set — so a future edit that reintroduces an invalid color
name fails the test suite immediately instead of waiting for another live demo to catch it.

**Verification performed**: confirmed Streamlit's valid-color list directly from its installed source
(`streamlit/elements/markdown.py`'s own docstring) rather than trusting memory; `pytest tests/ -v` — 27/27
passing; manually re-generated the exact markdown strings Streamlit will now render for all four risk
levels and confirmed each uses a color Streamlit actually supports.

## 2026-09-13 — Milestone 8: V3 — a real generalization failure, found live, fixed with transfer learning

**What happened**: while testing the app to prepare it for showing to friends/customers, a set of
genuinely natural example sentences (not phrased like the synthetic training templates) all came back
as MEDIUM risk regardless of actual severity — including sentences that were obviously critical
("the mobile app crashes every single time... I've already switched to a competitor") and obviously
positive ("the redesigned onboarding flow is fantastic"). This was **not** the color-rendering bug from
earlier the same day; it was the underlying urgency *score* itself failing to discriminate.

**Root cause, confirmed by direct measurement, not guessed**: V2's BiLSTM (`src/dl_lstm.py`) learns its
own embedding table from scratch, and its vocabulary — built only from words seen ≥2 times in the
~2,800-row training split — is **196 words total**. Tokenizing the stress-test sentences against that
vocabulary showed 33-63% of words mapping to `<UNK>`:

| Sentence type | OOV rate | V2 score |
|---|---|---|
| Matches training templates ("App keeps crashing during checkout payment step...") | 0% | 0.815 |
| Natural critical phrasing ("mobile app crashes every single time... switched to a competitor") | 55% | 0.393 |
| Natural positive phrasing ("redesigned onboarding flow is fantastic") | 42% | 0.428 |

With roughly half the words invisible to the model, V2 has almost no real signal to work with and its
output collapses toward the training mean — **critical and positive text scored 0.02 apart on average**,
regardless of true severity. This is a materially bigger problem than the "ambiguous boundary case"
limitation already documented in `DATA_INSIGHTS.md` — it's a full generalization failure on ordinary
human phrasing, not a hard edge case.

**Fix — V3, a transfer-learned regressor** (`src/transformer_regressor.py`, `scripts/train_v3.py`):
fine-tuned `distilbert-base-uncased` instead of training an embedding table from scratch. Froze the
embeddings and the first 4 of 6 transformer layers; trained only the last 2 layers + a small regression
head — **21.4% of parameters trainable**, deliberately mirroring the "fine-tune ~20% of a pretrained
backbone" pattern from a prior image-classification project, applied here to text. Subword tokenization
means there is no closed-vocabulary problem at all: any English word decomposes into known subword
pieces, so the model carries genuine pretrained language understanding into a dataset of only 4,000 rows
instead of having to learn English from those 4,000 rows.

**Also fixed as part of this**: V2's aggressive regex cleaning (strips punctuation/casing) was built for
a tiny from-scratch vocabulary where punctuation was just noise. A pretrained tokenizer benefits from
real casing and punctuation as signal, so V3 trains on `raw_text` rather than V2's `cleaned_text` —
different architectures warrant different preprocessing, not one cleaning pipeline for everything.

**Results — same held-out synthetic test set** (comparable to V2, confirming V3 didn't regress on what
V2 already did fine): test MSE 0.0160 (V2: 0.0162), test macro-F1 0.825 (V2: 0.821).

**Results — the stress test that exposed the bug** (`scripts/stress_test_compare.py`, same 8 sentences,
both models, side by side):

| | V2 (BiLSTM) | V3 (DistilBERT) |
|---|---|---|
| Mean score, CRITICAL examples | 0.44 | 0.60 |
| Mean score, LOW examples | 0.42 | 0.30 |
| **Critical − Low separation** | **0.022** | **0.298** |

V3's separation is **~13x larger** than V2's on text neither model was trained on verbatim — this is the
number that matters, not the synthetic test-set metrics, since the synthetic test set was already
in-distribution for both.

**App integration**: added `v3_transformer` as a third variant in `src/variants.py` (shares V2's dataset/
warehouse, has its own `model_dir`), a unified `src/urgency.py` dispatcher so `pages/1_Inference_Playground.py`
and `pages/3_GenAI_Copilot.py` work with either backend without knowing which one a variant uses, and a
new V2-vs-V3 section on the Model Comparison page with the real stress-test numbers above, visualized.
V3 is now the app's default variant.

**Verification performed**: `pytest tests/ -v` — 31/31 passing (added `tests/test_urgency_dispatch.py` and
updated `tests/test_variants.py` for 3 variants); a dedicated smoke-test agent confirmed all 3 variants
load correctly through the real dispatcher (not just standalone scripts), confirmed the exact critical/
positive separation numbers above via the actual app code path, and confirmed the live Streamlit server
still boots cleanly with no tracebacks after the page rewrites.

**Honest scope note**: this fixes the urgency regressor specifically. The K-Means clustering and baseline
classifiers still use TF-IDF (also vocabulary-limited in the same way) — flagged as the natural next
target in `docs/ENHANCEMENTS.md`'s existing "OpenAI embeddings for semantic clustering" item, not yet done.

## 2026-09-13 — Milestone 9: GitHub push warning, and finding V3's actual CRITICAL threshold

**GitHub upload issue**: the first V3 checkpoint saved the model's *full* state dict — 265MB — because
`torch.save(model.state_dict(), ...)` persists every parameter regardless of `requires_grad`. GitHub's
hard per-file limit is 100MB; this would have needed Git LFS for no real benefit, since ~78% of that
265MB is the *frozen*, unmodified DistilBERT backbone — identical to what `AutoModel.from_pretrained()`
already downloads fresh at load time (the same pattern this project already uses for BERT NER/BART).
**Fix**: added `TransformerUrgencyRegressor.trainable_state_dict()`, which filters the state dict down to
only the parameters with `requires_grad=True` (the last 2 of 6 transformer layers + the regression head),
and changed `load_transformer_regressor()` to reconstruct the frozen backbone via `from_pretrained()` and
overlay the slim checkpoint with `load_state_dict(..., strict=False)`. Result: 56.7MB, safely under
GitHub's limit — verified byte-for-byte identical predictions on the same test sentences before and after
the change, so this was a storage optimization, not a retrain. `git push` still printed GitHub's 50MB
*recommended* (not hard) limit warning, which is expected and harmless at 54MB.

**Finding V3's actual CRITICAL threshold, empirically**: the stress-test sentences from Milestone 8 all
scored HIGH (0.56–0.79) on V3, not CRITICAL (≥0.8) — genuinely severe complaints, but not crossing the
top threshold. Rather than guess why, tested ~20 more real-world-style sentences directly against the
model. Two phrasings reliably cross 0.8:

| Score | Statement |
|---|---|
| 0.860 | "Your platform deleted all of my account data without any warning, I need this fixed immediately, this is completely unacceptable." |
| 0.847 | "Your system deleted all of my customer records without any warning, I need this fixed immediately, this is completely unacceptable and has put my entire business at risk." |
| 0.828 | "Your app deleted all of my data without any warning, I need this fixed right now, this is completely unacceptable and has put my business at serious risk." |
| 0.808 | "Someone accessed my account without my permission and changed my password, I am now completely locked out and my financial data is at serious risk." |
| 0.803 | "Someone accessed my account without permission and changed my password, I am locked out of everything and there is sensitive financial data at risk." |

**Pattern identified**: (1) *data deletion + demand for an immediate fix + "unacceptable"* language, and
(2) *unauthorized account access + password change + financial data at risk* — both map closely to a
specific training-template pattern in V2's hardened dataset ("Your {product} deleted all my data without
warning, I need this fixed NOW."). Severity alone (e.g., "production API down for 45 minutes, losing
revenue") reliably reaches HIGH but not CRITICAL.

**Honest caveat, stated plainly**: V3's CRITICAL threshold is calibrated to a fairly narrow severity
pattern learned from the training data's specific template families (data loss, account compromise) —
not a general "how bad does this sound" scale. A genuinely severe outage complaint that doesn't match
that narrower pattern will land HIGH, not CRITICAL. This is not a bug — thresholds calibrated to training
distribution are expected behavior — but it is a real limitation worth stating to anyone evaluating the
0.8 cutoff as if it meant "the model agrees this is as bad as it gets." Flagged as a concrete input for
`docs/ENHANCEMENTS.md`'s real-dataset validation item: training on genuinely diverse real complaint data
(not just V2's template families) would likely broaden what triggers CRITICAL beyond these two patterns.

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

## 2026-09-13 — Milestone 10: Real-data validation — V3's fix replicates outside the lab

**Why**: every V2-vs-V3 number in Milestone 8/9 came from 8 sentences I wrote myself. Even reported
honestly, that's a real scientific weakness — a skeptical reader can reasonably ask "did you just write
examples that make your fix look good?" Real, independently-authored text is the actual test.

**Method** (`scripts/validate_real_data.py`): fetched 5,000 Yelp Review Full rows and 5,000 Sentiment140
rows (`scripts/fetch_real_datasets.py`). Neither dataset has a native "urgency" label, so used the natural
proxy: Yelp 1-star reviews / Sentiment140 negative tweets as proxy-critical, Yelp 5-star / Sentiment140
positive as proxy-low. Sampled 150 per class per dataset, scored all of them with both V2 and V3, and
reused the exact "critical-minus-low mean score separation" metric from the hand-written stress test —
same methodology, real text, ~19x more examples per group than the original 4.

**Results**:

| | Yelp (n=150/class) | Sentiment140 (n=150/class) |
|---|---|---|
| V2 separation | 0.0249 (Mann-Whitney p=0.0048 — significant, but tiny) | 0.0050 (p=0.39 — **not significant**) |
| V3 separation | 0.1272 (p<0.0001) | 0.0837 (p<0.0001) |
| V2 vs. 5-class star rating | Spearman ρ=0.073 (p=0.14 — not significant) | — |
| V3 vs. 5-class star rating | Spearman ρ=0.682 (p<0.0001) | — |

**Honest read, stated plainly rather than rounded up**: V3's separation replicates on independent real
text and is statistically significant on both datasets — the core finding holds. But the *effect size* on
real data (0.08–0.13) is meaningfully smaller than the hand-written stress test's 0.30 — real reviews are
noisier and more mixed than text written to be unambiguous, and that gap is worth reporting rather than
letting the bigger, cleaner number stand unchallenged.

A genuinely interesting secondary finding: V2's separation is *not uniformly useless* — on Yelp (longer
reviews), it's small but statistically real (p=0.0048); on Sentiment140 (short tweets), it's statistically
indistinguishable from noise (p=0.39). Consistent with the Milestone 8 root cause: shorter text gives a
196-word vocabulary proportionally less to work with, so the failure is worse on tweets than on full
reviews. V3's Spearman correlation against Yelp's full 5-class rating (ρ=0.682, p<0.0001) is the strongest
single piece of evidence in the project that V3 tracks real sentiment — V2's equivalent (ρ=0.073, not
significant) shows essentially no real predictive relationship across the full ordinal scale.

**App integration**: added a "Real-World Validation" section to `pages/0_Model_Comparison.py` — two bar
charts (Yelp, Sentiment140) with Mann-Whitney p-values, the Spearman granular check, and the honest-read
callout above, so a viewer sees the nuance rather than a single flattering number.

**Verification performed**: `pytest tests/ -v` — 31/31 passing; a dedicated smoke-test agent confirmed the
new page section's exact data path (JSON loading, field access, p-value formatting) runs without
exceptions against the real `models/real_data_validation.json`, and that the live Streamlit server boots
cleanly with the new section.

## 2026-09-13 — Milestone 11: Clustering Explorer crashed on V3 — found by a screenshot, fixed properly

**Found by**: a screenshot of the live app showing `KeyError` on the Clustering Explorer page with V3
selected in the sidebar.

**Root cause**: V3 only replaces the urgency regressor (BiLSTM → DistilBERT); clustering (TF-IDF +
K-Means) was never retrained separately for it — V3's `training_summary.json` has urgency-training keys
(`test_mse`, `stopped_epoch`, ...) but no `best_k`/`k_sweep_scores`, and `models/v3_transformer/` has no
`tfidf.pkl`/`kmeans_model.pkl` at all. The page unconditionally read `variant["model_dir"]`, which broke
the moment a variant without clustering artifacts existed.

**Decision**: keep the page (V1-vs-V2 clustering/elbow comparison is core to the project's story) rather
than remove it, and make V3 fall back to V2's clustering artifacts explicitly, with a banner explaining
why, instead of silently erroring or silently duplicating V2's charts unlabeled. `pages/2_Clustering_Explorer.py`
now resolves a `clustering_variant` (V2 Hardened, whenever the selected variant's `model_type` is
`"transformer"`) and uses that for the summary, warehouse, and artifact lookups, while the sidebar still
shows the actual selected variant.

**Verification performed**: `pytest tests/ -v` — 31/31 passing; confirmed the pre-fix `KeyError` precondition
directly (`v3_transformer`'s summary keys have no `best_k`); a smoke-test agent replicated the page's exact
logic for all 3 variants against the live warehouse/artifacts and confirmed no exception and a populated
cluster assignment for each; live Streamlit server boots cleanly.

## 2026-09-13 — Milestone 12: A "V3 superiority scorecard" — feedback that V2 vs. V3 looked identical

**Feedback received**: after using the app directly, the difference between V2 and V3 wasn't obvious
enough when clicking between them on ordinary pages — the case for V3 lived mostly in prose and separate
charts, not as one unmissable, quantified comparison. Also flagged: the app itself never disclosed the
real-data validation (Milestone 10) — a user or interviewer evaluating the live app had no way to know
V3 had been checked against real Yelp/Sentiment140 data at all.

**Built** (`scripts/superiority_scorecard.py`): a consolidated scorecard combining every existing
V2-vs-V3 metric (hand-written separation, Yelp separation, Sentiment140 separation, Yelp Spearman
correlation, synthetic test MSE/F1) into one table with an explicit win/lose call per metric, plus two
new concrete artifacts:

1. **V2 tested on the 5 known CRITICAL-triggering examples for the first time** (Milestone 9 only ran V3
   on these). Result: 2 of the 5 produce a genuine 2-tier jump — **"Someone accessed my account without
   permission and changed my password..."** scores **V2: MEDIUM (0.47) → V3: CRITICAL (0.81)** — the
   direct answer to "which examples does V3 correctly flag as CRITICAL that V2 does not." The other 3
   (data-deletion phrasing) turned out to closely match a V2 training template, so V2 also scores them
   CRITICAL/HIGH — reported honestly as *not* flip examples rather than cherry-picked to inflate the count.
2. **Risk-tier flips on the original 8-sentence stress test**: 4 of 8 sentences change tier entirely
   between V2 and V3 (not just a score nudge) — 2× MEDIUM→HIGH, 2× MEDIUM→LOW.

**App changes**:
- `pages/0_Model_Comparison.py`: new prominent section at the top of the V2-vs-V3 area — a "validated on
  real data" success banner, a full metrics scorecard table (winner marked, significance shown), and the
  concrete flip examples rendered as direct side-by-side text; the existing stress-test table now also
  shows computed risk tiers and a "tier changed?" column instead of only raw scores.
- `pages/1_Inference_Playground.py`, `pages/3_GenAI_Copilot.py`: added a persistent banner whenever V3 is
  the active variant, stating the real-data validation result directly where a user is actually scoring
  text — not just on the comparison page they might not visit.

**Verification performed**: `pytest tests/ -v` — 31/31 passing; a smoke-test agent replicated the exact
new page logic (metric dict formatting, flip-example filtering, risk-tier computation and column rename)
against the real JSON files and confirmed the numbers (3 critical-trigger flips, 4/8 stress-test tier
flips) match what's in the report; live Streamlit server boots cleanly with no tracebacks.

## 2026-09-13 — Milestone 13: Executive summary barely changed between n=17 and n=46 — diagnosed, not guessed

**Found by**: a screenshot comparing the Batch Executive Summary at slider values 17 and 46 — the output
was almost identical (3 of 4 sentences shared verbatim).

**Diagnosis, confirmed against real data before touching any code**: queried the warehouse directly.
Among the top 46 most-urgent records by `urgency_score`, only **30 were unique raw text** — 16 were exact
duplicates of already-included sentences with only filler words changed (e.g., "Been on hold for an
hour/two hours/45 minutes trying to reach a human" counted 4 times). Root cause: the synthetic generator
draws critical feedback from only 8 template families (`scripts/generate_data.py`), so as N grows past
the low teens, most "new" records the slider adds are near-duplicates of ones BART already saw — the
summarizer was correctly condensing what it was given, but what it was given had far less real information
than its row count implied. Confirmed by counting unique texts at several N: n=5→5 unique, n=17→12,
n=46→30, n=100→43, plateauing around 212 total unique texts in the full dataset (a familiar number — the
same 212-unique-sentence figure from the V1-vs-V2 story, since urgency ranking pulls disproportionately
from the same template family repeatedly at the high end).

**Fix**: `pages/4_NER_and_Summarization.py` now deduplicates on `raw_text` *before* slicing the top-N,
so the slider controls how many **distinct** complaints reach BART rather than how many rows (duplicates
included). Added a caption showing how many unique critical texts exist in the dataset, so the behavior
is transparent rather than mysterious. Verified directly: with dedup, n=17 and n=46 now feed genuinely
different sentences and produce genuinely different summaries.

**Honest residual behavior, documented rather than hidden**: even after the fix, a larger N can
occasionally produce a *shorter* summary than a smaller N. This is because `summarize_batch` chunks input
over BART's ~1024-token limit and, when more than one chunk is produced, summarizes the chunk-summaries
once more to keep the final brief a fixed length regardless of batch size (see `src/transformer_nlp.py`
docstring) — that second compression pass can drop a theme a single-chunk summary would have kept. This
is expected behavior for hierarchical/recursive summarization, not a new bug, but worth stating so a
future test of this page isn't mistaken for a second unresolved issue.

**Verification performed**: `pytest tests/ -v` — 31/31 passing; directly queried the warehouse to confirm
the duplication root cause (30/46 unique) before writing any fix; re-ran `summarize_batch` on the deduped
top-17 vs. top-46 lists and confirmed the two summaries now differ in genuinely new content rather than
being coincidentally near-identical.

## 2026-09-13 — Milestone 14: "This app has gone over its resource limits" — real memory overflow in production

**Found by**: the live deployed app, directly — Streamlit Cloud's own out-of-memory page ("It's using too
much memory!"), not a code exception. Initially misdiagnosed from the logs alone as the already-fixed
Clustering Explorer `KeyError` (a stale log the user re-pasted); the actual current failure only became
clear from a screenshot of the real error page.

**Root cause, measured, not estimated**: queried HuggingFace's file metadata directly for the two
heaviest models the app was loading — `dbmdz/bert-large-cased-finetuned-conll03-english` (1,334MB) and
`facebook/bart-large-cnn` (1,625MB), **2,959MB combined**, on top of V3's DistilBERT (~260MB in memory
once loaded, even though its checkpoint on disk is only 56.7MB — see Milestone 8) plus two small BiLSTMs.
Streamlit Community Cloud's free tier has a hard memory ceiling; adding V3 on top of an already-heavy
BERT-large + BART-large combination pushed a previously-marginal memory budget over the edge.

**Fix**: swapped both models for meaningfully smaller equivalents doing the same job —
`dslim/bert-base-NER` (433MB, same CoNLL03 PER/ORG/LOC/MISC tag scheme) and `sshleifer/distilbart-cnn-12-6`
(1,222MB, a distilled BART fine-tuned for the same CNN/DailyMail-style summarization task). Combined:
1,655MB — a **1.3GB reduction**.

**Quality verified, not assumed**: re-ran the exact same test sentence used since Milestone 2
("The iPhone 15 crashed after the iOS 17.2 update...") — all 4 entities still extracted correctly at
>99% confidence (iPhone 15, iOS, Apple, California), though the smaller NER model split "iOS 17.2" into
just "iOS" rather than the full version string — a minor precision loss, disclosed rather than hidden.
Summarization re-tested at realistic production batch sizes (the same deduped top-17 and top-46 lists
from Milestone 13): genuine compression held (1,176→244 chars at n=17, 3,086→276 chars at n=46) and the
two summaries remained genuinely different from each other, confirming the model swap didn't undo the
Milestone 13 dedup fix.

**Verification performed**: `pytest tests/ -v` — 31/31 passing (no test depended on the specific model
names); live-tested both new models with real inference before committing, at both toy scale and
production batch scale; updated the one user-facing caption that named the old models.

**Immediate mitigation given to the user**: reboot the app via Streamlit Cloud's dashboard to clear the
already-overflowed memory while this fix deploys.

**Follow-up**: shortly after, Streamlit Cloud separately throttled the app's CPU ("we've temporarily
reduced its CPU to keep the platform healthy for everyone"), expiring at a stated time. This is a
different mechanism from the memory error above — a free-tier CPU cap triggered by sustained heavy
compute, not an error state — and is expected given this app runs 4 transformer-based models (BERT-NER,
a summarizer, and V3's DistilBERT) with no GPU, compounded by a day of repeated redeploys and testing
against the same free-tier instance. No code fix applies here; documented as an honest limitation of
free-tier hosting for a project this compute-heavy, not glossed over. The model-size reduction above
should help some (smaller models = fewer CPU-seconds per inference call), but a project like this
genuinely strains what a free tier is designed for — see `docs/PRODUCTION_READINESS.md` for the paid-tier
discussion this motivates.

## 2026-09-13 — Milestone 15: Executive summaries "read like copy-pasted tickets" — an architectural mismatch, not a tuning problem

**Feedback**: the Batch Executive Summary lacked genuine synthesis — it read as reused ticket sentences
rather than a real executive brief identifying cross-cutting themes.

**Diagnosis**: BART/DistilBART-family models are trained on CNN/DailyMail news articles to do
span-compression — a well-documented property in the summarization literature is that these models
behave close to extractive (copying/lightly editing salient source sentences) even when technically
"abstractive." That's a workable approximation of "summary" for long flowing news prose, but our tickets
are already terse, keyword-dense complaint sentences ("Major outage across the platform, unable to log
in for hours.") — there's no flowing prose to compress, so the model's span-selection behavior surfaces
directly as "it just copied the tickets." No BART-family model size swap fixes this: it's a mismatch
between the tool (span-compression) and the actual task (identify recurring themes across many short,
independent statements and write new synthesizing prose) — the same "model class doesn't match the task"
pattern as Milestone 8's BiLSTM vocabulary problem, just for a different component.

**Fix**: extended `generate_resolution()`'s existing pluggable-backend pattern (Claude if configured,
deterministic local fallback otherwise) to the executive summary feature. `generate_executive_summary()`
in `src/genai_copilot.py` uses Claude with an explicit synthesis-oriented system prompt ("identify the
2-4 recurring themes... do NOT simply copy, lightly reword, or concatenate individual ticket sentences")
when `ANTHROPIC_API_KEY` is configured, and falls back to the existing BART pipeline (still useful, just
honestly labeled as more extractive) with zero configuration required otherwise. The app now shows which
backend produced a given summary and, on the local fallback, explains why it reads extractive rather than
hiding the limitation.

**Honest limitation of this verification**: no Anthropic API key is available in this development
environment, so the Claude synthesis path's actual *output quality* could not be tested end-to-end here
— only its plumbing (backend selection, JSON/dict shape, UI branching) was verified, via a monkeypatched
key and canned response. The local BART fallback path was fully verified against real data and confirmed
unchanged from its prior behavior. The user will need to add `ANTHROPIC_API_KEY` and test the Claude path
themselves to confirm the synthesis quality improvement actually lands as intended.

**Verification performed**: `pytest tests/ -v` — 31/31 passing; live Streamlit smoke test confirmed the
local-fallback path runs end-to-end against the real warehouse with no exceptions and produces the same
summary as before the change (backend correctly reported as `"local_bart"`); a monkeypatched test
confirmed the Claude code path's structure (dict shape, UI branch) without exercising the real API.

## 2026-09-13 — Milestone 16: NER tags "iPhone 15" / "iOS" as MISC — a tagging-scheme limit, not a model bug

**Feedback**: user asked why the NER page tags product/software names ("iPhone 15", "iOS") as `MISC`
instead of something more specific, and whether it's fixable.

**Diagnosis**: both the prior model (`dbmdz/bert-large-cased-finetuned-conll03-english`) and the current
one (`dslim/bert-base-NER`, swapped in Milestone 14 for memory) are trained on **CoNLL-2003**, which
defines only 4 entity types: `PER`, `ORG`, `LOC`, `MISC`. There is no `PRODUCT` or software-version tag
in this scheme at all — `MISC` is the designated catch-all for anything that isn't a person, org, or
place (nationalities, events, products, software). So this is not a regression from the Milestone 14
model swap; both models share the identical tag set and would behave the same way here.

**Investigated a real fix**: an OntoNotes-5-trained model would add a dedicated `PRODUCT`/`WORK_OF_ART`
tag. Checked two candidate small HF repos (`tner/roberta-base-ontonotes5`, `tner/deberta-v3-base-ontonotes5`)
via the HF API — both 404, not reliably available. `Babelscape/wikineural-multilingual-ner` (709MB) is
available but still uses the same 4-class CoNLL-style scheme, so it wouldn't fix anything. The one real
OntoNotes option, `flair/ner-english-ontonotes-large`, is a different library (Flair, not `transformers`)
at ~1.7GB — larger than our entire current NER+summarizer footprint combined (1.66GB, Milestone 14), and
would reintroduce the exact memory pressure just fixed.

**Decision**: not worth it at this project's scale — no small, reliable, `transformers`-pipeline-native
model with a PRODUCT tag exists, and the one real alternative would undo the recent memory fix. Fixed the
*honesty* of the page instead: added an inline caption on the NER page, shown whenever a MISC entity
appears, explaining the CoNLL-2003 scheme limitation and why a fix was investigated and passed on —
documented as a known limitation rather than silently left unexplained.

**Verification performed**: confirmed both models' shared CoNLL-2003 provenance; verified via
`huggingface_hub.HfApi().repo_info(files_metadata=True)` that the OntoNotes `tner` candidates don't
resolve and that `wikineural` is 709MB but not OntoNotes-schemed; live-read the updated
`pages/4_NER_and_Summarization.py` caption logic.
