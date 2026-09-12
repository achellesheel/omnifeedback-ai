# Product Review: End-to-End Flow, QA Findings & Business Justification

A QA/PM-style walkthrough of OmniFeedback AI as it exists today (live at
https://omnifeedback-ai-c4mmclqnkgytms6vuzqtjp.streamlit.app/), written against the actual shipped
product rather than the original spec's aspirational description.

## 1. End-to-end user journeys

### Journey A: Support manager triaging a single incoming ticket
1. Opens **Inference Playground**, pastes the raw ticket text.
2. Gets an urgency score (0.0–1.0) in under a second, plus a plain-language risk label
   (LOW/MEDIUM/HIGH/CRITICAL).
3. If CRITICAL/HIGH, moves to **GenAI Copilot**, pastes the same text, gets a root-cause guess,
   recommended action, and a draft customer reply in ~1–2 seconds (local backend) or a few seconds
   (Claude backend).
4. **Friction point (real)**: the two pages don't share state — the manager has to re-paste the same
   text into the Copilot page after checking urgency in the Playground. Low-effort fix noted in
   `ENHANCEMENTS.md`.

### Journey B: Product manager reviewing weekly feedback trends
1. Opens **Executive Overview** — sees critical rate, urgency-by-channel box plot, aspect volume, and a
   time-series trend with an escalation-threshold line.
2. Opens **Clustering Explorer** to see which text patterns are driving volume, with the elbow chart
   right there to justify why K was chosen the way it was (not just asserted).
3. Opens **Statistical Tests** to check whether channel is statistically associated with urgency —
   currently returns p=0.84 on the shipped data (see `DATA_INSIGHTS.md` §5), which is an honest "no" a
   PM should not casually override with "well it seems like Twitter is worse" intuition.
4. **Gap (real)**: there's no saved/exportable report — a PM has to screen-share the live app rather than
   generate a PDF/shareable snapshot. Noted in `ENHANCEMENTS.md`.

### Journey C: Skeptical technical reviewer (recruiter, hiring manager, conference reviewer)
1. Opens **Model Comparison** first (it's page 0, first in the sidebar).
2. Sees V1 naive hit F1=1.0 / silhouette=1.0, sees the "why this is a red flag" callout, sees the elbow
   curve overlay, sees the actual sample text from both variants.
3. This journey exists specifically because **a metric alone is not a credible proof of competence** —
   a viewer who has been burned by inflated capstone-project claims before can verify the claim
   themselves in under a minute. This is the strongest single feature for the "get noticed" goal behind
   this whole build.

## 2. QA findings — issues identified and their current status

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Original synthetic generator (V1) made the whole pipeline trivially overfit-prone | High | **Fixed** — V2 hardened generator, documented in `PROGRESS.md` Milestone 1 |
| 2 | K-selection via naive `argmax(silhouette)` picks the range boundary on templated text | Medium | **Fixed** — Kneedle elbow method, `PROGRESS.md` Milestone 1 |
| 3 | Docker `HEALTHCHECK` silently broken (missing `curl` in slim image) | Medium | **Fixed** — `PROGRESS.md` Milestone 4 |
| 4 | Streamlit Cloud defaults to a Python version incompatible with pinned `torch` | High (blocks deploy) | **Fixed** — `runtime.txt` + Advanced Settings, `PROGRESS.md` Milestone 5 |
| 5 | `channel`/`aspect_category` carry no real signal in the synthetic data (independent of text) | Medium | **Open** — documented in `DATA_INSIGHTS.md` §5, fix proposed in `ENHANCEMENTS.md` |
| 6 | Inference Playground and GenAI Copilot don't share pasted text between pages | Low | **Open** — `ENHANCEMENTS.md` |
| 7 | No exportable/shareable report snapshot for the Executive Overview page | Low | **Open** — `ENHANCEMENTS.md` |
| 8 | NER/BART cold-start is ~95s combined on first use (multi-GB model download) | Low (UX, not correctness) | **Open** — acceptable for a demo, flagged for a production SLA discussion |
| 9 | GenAI local backend's root-cause detection is keyword-pattern-based, not learned | Medium | **By design** — it's the zero-cost fallback; Claude backend is the upgrade path (already wired) |

## 3. Business value — why this matters to a client, mapped to the platform's own claims

The original spec's business cases (real-time crisis escalation, aspect clustering for roadmap
prioritization, executive summarization, GenAI-assisted resolution drafting) are all **actually
implemented and running**, not aspirational — which is itself the main thing a technical buyer should
verify before trusting any of the following ROI framing:

- **Crisis detection speed**: the BiLSTM scores a ticket in well under 350ms (per `PROGRESS.md`
  Milestone 1's latency measurement of 0.025ms/record for ingestion; live BiLSTM inference in the
  smoke tests ran in well under 100ms per call). At the 0.5 threshold, it catches ~80% of true critical
  tickets in `DATA_INSIGHTS.md`'s confusion matrix — a real, honestly-reported number, not a marketing
  claim of near-100%.
- **Clustering for roadmap prioritization**: works, but its business value is currently capped by
  Finding #5 above — until channel/aspect carries real signal (via real data or a generator fix), the
  clusters describe *how people phrase feedback*, not *what product area it's about*. This is an
  important caveat to give a prospective client rather than oversell.
- **Executive summarization**: BART live-tested and confirmed coherent (Milestone 2) — genuinely reduces
  a batch of tickets to a 1–2 sentence brief, which is the actual time-savings claim, verified rather
  than assumed.
- **GenAI resolution copilot**: works with zero API cost via the local backend, upgrades transparently to
  Claude if a client wants materially better draft quality — this tiered design directly addresses the
  common enterprise objection "we don't want per-token costs during a pilot."

## 4. Recommendation

The platform is credible to demo to a technical audience *today*, with two caveats a presenter should
state proactively rather than have discovered: (1) the clustering/channel-association findings are
currently constrained by synthetic-data design, not model capability, and (2) the GenAI local backend is
intentionally simple — its value is the zero-cost default and clean upgrade path, not root-cause
sophistication. Both caveats are already honestly documented rather than hidden, which is itself part of
the pitch for a technically literate buyer.
