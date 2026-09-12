# Production Readiness Checklist

`ENHANCEMENTS.md` covers feature/technique ideas. This document is narrower and more operational:
what would actually need to change for this to run as real production infrastructure serving live
traffic, rather than a portfolio demo — ranked by what would break first.

## What would break first, in order

1. **SQLite under concurrent write load.** SQLite locks the whole database file per write
   transaction. `WarehouseManager.insert_feedback_batch` already wraps writes in one transaction
   (good), but multiple concurrent ingestion workers would serialize and eventually queue up. Fix:
   migrate to PostgreSQL only when real concurrent-write volume is confirmed — premature before that
   (see `ENHANCEMENTS.md`'s "explicitly not recommended right now" section, which still holds).
2. **No authentication or rate limiting on the Streamlit app.** The live demo is intentionally public
   for portfolio purposes, but a real deployment handling real customer feedback needs at minimum
   Streamlit's built-in auth (via `st.experimental_user` / SSO on paid tiers) or a reverse proxy with
   auth in front of it, plus rate limiting on the GenAI Copilot page specifically — an unauthenticated
   page that calls a paid LLM API per click is a direct cost-abuse vector the moment `ANTHROPIC_API_KEY`
   is configured.
3. **Secrets handling is demo-grade.** `.streamlit/secrets.toml` works for Streamlit Cloud but has no
   rotation, no audit log, and no scoping. Production: a real secrets manager (AWS Secrets Manager /
   GCP Secret Manager / HashiCorp Vault), rotated on a schedule, injected as environment variables at
   deploy time rather than a committed-adjacent file.
4. **No CI pipeline.** Tests exist (25 passing) but nothing runs them automatically on push/PR. Fix:
   a GitHub Actions workflow running `pytest tests/` on every push, blocking merge on failure — this is
   the single highest-leverage, lowest-effort item on this entire list and should be done regardless of
   deployment target.
5. **No structured logging or error monitoring.** Right now, a production failure (a malformed GenAI
   response, a corrupt CSV upload, a model file going missing) surfaces only as whatever Streamlit
   prints to its own console. Fix: structured logging (Python `logging` with JSON formatting) plus a
   hosted error tracker (Sentry has a workable free tier) wired into `app.py` and each page.
6. **No model/data versioning beyond folder naming.** `models/v1_naive/` and `models/v2_hardened/` are
   a manual, ad-hoc versioning scheme — fine for two known variants, not fine for a real MLOps loop
   where models get retrained on a schedule. Fix: something like DVC or MLflow's model registry once
   there's a real retraining cadence to manage.
7. **No automated retraining or drift detection.** The BiLSTM and TF-IDF/K-Means artifacts are static
   files trained once and committed. In production, the input distribution will drift from whatever
   `scripts/generate_data.py` produced. Fix: scheduled retraining job + the drift-monitoring idea already
   in `ENHANCEMENTS.md`, but this only matters once there's a live feedback stream to actually drift.
8. **Health checks exist for Docker, not for the app's actual dependencies.** The `HEALTHCHECK` (fixed
   in Milestone 4) confirms the Streamlit process is up — it does not confirm the warehouse is reachable,
   the model files loaded successfully, or the GenAI backend is responding. Fix: an internal
   `/healthz`-equivalent check (Streamlit doesn't expose custom routes easily, so this would likely mean
   a small sidecar health endpoint, or moving health logic into the Docker `HEALTHCHECK` script itself
   to actually import and sanity-check the model artifacts, not just hit `/_stcore/health`).
9. **No load testing.** The 0.025ms/record ingestion latency and sub-350ms target are measured on a
   local single-process run, not under concurrent load. Fix: a basic Locust or k6 script hitting the
   ingestion path and BiLSTM inference concurrently, before any claim about production throughput.

## What's already in reasonably good shape

- **ACID transaction discipline** in `WarehouseManager` — rollback-tested, not just asserted.
- **Reproducible builds**: pinned `requirements.txt`, `Dockerfile`, `runtime.txt` — verified to build and
  run correctly (Milestone 4), including catching a real healthcheck bug before it shipped.
- **Graceful degradation** in the GenAI copilot: the app never hard-fails to a missing API key, it
  degrades to a deterministic local backend. This is a production-relevant pattern already implemented
  for the right reason (a public demo can't assume secrets are configured), not just for cost reasons.
- **Test coverage of failure paths**, not just happy paths — `test_rollback_on_bad_urgency_score` and
  `test_missing_required_column_raises` specifically test what happens when things go wrong, which is
  the part of a test suite that's usually skipped under time pressure.

## Recommended order if this were actually going to production

1. CI pipeline (GitHub Actions running pytest) — hours of effort, prevents regressions immediately.
2. Structured logging + Sentry — a few hours, makes every subsequent bug 10x faster to diagnose.
3. Auth + rate limiting in front of the GenAI Copilot page specifically — required before enabling a
   paid API key in any public-facing deployment.
4. Secrets manager migration — required before handling any real customer data, not just nice-to-have.
5. Everything else (Postgres migration, MLOps registry, drift monitoring, load testing) — only once
   there's a real usage pattern to size those investments against. Building them speculatively now would
   be exactly the kind of premature complexity this project's `ENHANCEMENTS.md` already argues against
   for the Kafka/Postgres case.
