# Enhancement Roadmap

Ideas ranked by cost/effort vs. impact, grounded in the specific gaps found in `DATA_INSIGHTS.md` and
`PRODUCT_REVIEW.md` — not a generic wishlist.

## Free / low-effort (do these first)

1. **Fix channel/aspect independence in the synthetic generator** (`DATA_INSIGHTS.md` §5). Make
   `scripts/generate_data.py` correlate `channel` and `aspect_category` with the sampled template
   category (e.g., `App Crash` templates skew toward `Mobile App`/`Twitter`; `Billing Defect` skews
   toward `Email Support`/`Call Center Note`). This turns the chi-squared test from "correctly detects
   manufactured independence" into "correctly detects a real, designed association" — a more convincing
   demo of the statistics page and gives clustering a real aspect signal to recover.
2. **Share state between Inference Playground and GenAI Copilot** (`PRODUCT_REVIEW.md` #6) via
   `st.session_state` — paste once, reuse everywhere. Small UX win, ~20 lines of code.
3. **Threshold calibration control**: expose the 0.48-vs-0.5 threshold finding from `DATA_INSIGHTS.md`
   §2 as a slider on the Inference Playground / Executive Overview, so a user can trade precision for
   recall live instead of the app hard-coding 0.5.
4. **PDF/HTML export of the Executive Overview** (`PRODUCT_REVIEW.md` #7) — Streamlit can render a page
   to static HTML via `st.download_button` + a Plotly `fig.to_html()` bundle; no new dependency needed.
5. ~~**Real-dataset validation pass**~~ — **done** (2026-09-13, PROGRESS.md Milestone 10): ran V2 and V3
   against 150 examples/class of real Yelp reviews and Sentiment140 tweets. V3's separation replicated,
   significantly, on both (p&lt;0.0001), at roughly half the hand-written stress test's magnitude — the
   fix holds up, honestly reported at its real effect size. Now live on the app's Model Comparison page
   and in `docs/DATA_INSIGHTS.md` §7. Natural follow-up: extend the same real-data validation to the
   TF-IDF-based clustering/classifiers (item below), which haven't received it yet.

## Paid APIs (cheap tier, high leverage)

| Option | What it adds | Rough cost | Where it plugs in |
|---|---|---|---|
| **Anthropic Claude API** (already wired) | Materially better CoT reasoning + reply drafts | ~$3/M input tokens (Sonnet) | `src/genai_copilot.py` — just add `ANTHROPIC_API_KEY` to Streamlit secrets, zero code change |
| **OpenAI text-embedding-3-small** | Semantic (not just TF-IDF) clustering + "find similar past tickets" search | ~$0.02/M tokens — effectively free at this dataset's scale | New clustering path in `src/ml_models.py`; would likely raise silhouette meaningfully since embeddings capture semantics, not just keyword overlap |
| **Pinecone / pgvector** | Persistent vector index for "similar historical ticket" retrieval in the GenAI Copilot (RAG) | Free tier covers this dataset's scale | Copilot could cite 2-3 similar past resolutions before drafting a reply — directly improves consistency, a named goal in the original business case |
| **Twilio / Slack webhook** | Real automated escalation alert (not just a UI banner) when urgency crosses threshold | Pennies per alert | Hook into `WarehouseManager.insert_feedback_batch` or a scheduled job |
| **HuggingFace Inference Endpoints (paid tier)** | Moves BERT NER / BART off the free CPU-only Streamlit Cloud tier, cutting the ~95s cold-start (`PRODUCT_REVIEW.md` #8) | ~$0.05–$0.10/hr for a small GPU endpoint, pay-as-you-go | Swap `src/transformer_nlp.py`'s local pipeline calls for HTTP calls to a hosted endpoint |

## Bigger technique bets (for the research-paper phase)

1. **Fine-tune a small transformer (DistilBERT) directly for urgency regression**, as a second deep
   learning approach alongside the BiLSTM. This is the natural "second algorithm" for the eventual paper's
   comparison table — same task, transformer vs. recurrent architecture, both evaluated identically via
   `scripts/data_insights.py`'s methodology.
2. **Active learning loop**: use the BiLSTM's prediction confidence (distance from 0.5) to flag the most
   ambiguous records for a human-in-the-loop relabel pass — directly targets the false-negative pattern
   found in `DATA_INSIGHTS.md` §4 (most errors are boundary cases).
3. **Model/data drift monitoring**: track the urgency-score distribution and clustering silhouette over
   time in production; alert if either drifts significantly, since that's the realistic failure mode for
   a deployed model (the training distribution stops matching live traffic) rather than sudden accuracy
   collapse.
4. **Multi-lingual support**: swap the BERT NER/BART models for multilingual variants
   (`xlm-roberta`-based), since real enterprise feedback is rarely English-only — directly extends the
   platform's addressable business case.

## Explicitly not recommended right now

- **Switching the warehouse to Postgres**: no current bottleneck justifies it; SQLite handles this
  dataset's scale (and Streamlit Cloud's free tier) fine. Revisit only if a real multi-tenant deployment
  is actually planned.
- **Kafka/streaming ingestion**: massive infra overkill for a portfolio/demo project at this stage;
  worth mentioning in a research paper as a "how this would scale" discussion, not worth building now.
