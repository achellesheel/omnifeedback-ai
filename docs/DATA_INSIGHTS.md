# Data Insights — V2 (Production) Model

Computed by `scripts/data_insights.py` against the V2 Hardened variant (4,000 records). Every number
below is reproducible by running that script — nothing here is estimated or rounded up for effect.

## 1. Classification performance at the deployed threshold

| | Predicted Non-Critical | Predicted Critical |
|---|---|---|
| **Actually Non-Critical** | 2,300 (TN) | 355 (FP) |
| **Actually Critical** | 275 (FN) | 1,070 (TP) |

- **Macro-F1 @ 0.5 threshold: 0.826** (matches `training_summary.json`'s held-out test-set F1 of 0.821 —
  consistent across the full dataset and the test split, which is itself a good sign: no overfitting
  to the test split specifically).
- **Precision (critical class): 75.1%** — 1 in 4 flagged-critical tickets is a false alarm.
- **Recall (critical class): 79.6%** — roughly 1 in 5 truly critical tickets is missed at this threshold.

## 2. Threshold tuning has limited headroom — and that's a good sign

Swept macro-F1 across thresholds from 0.05 to 0.95 in steps of 0.005:

- **Best threshold found: 0.48** (barely left of the default 0.5)
- **Best macro-F1: 0.8285** vs. **0.826 at the default** — a **0.3-point** improvement.

This is a deliberately small gain, and that's informative: it means the model's default 0.5 cutoff is
already close to optimal, so there was no hidden accuracy being left on the table by using the "obvious"
threshold. (An earlier draft of this script accidentally optimized a *different, non-comparable* metric —
sklearn's `precision_recall_curve` only tracks positive-class F1, not macro-F1 — and it found a "best"
threshold that actually scored *worse* than 0.5. Caught by noticing the numbers didn't make sense and
re-deriving the sweep to match the metric used everywhere else in the project. See `PROGRESS.md` for the
full account — worth keeping as a cautionary example about comparing metrics that aren't the same metric.)

## 3. Predicted vs. true urgency correlation: 0.869

Pearson correlation between the BiLSTM's continuous output and the ground-truth `urgency_score` is
**0.869** — strong linear agreement, consistent with the low test MSE (0.016) reported in Milestone 1.
The model isn't just getting the binary critical/non-critical call right; it's tracking the *magnitude*
of urgency reasonably well too, which matters for the escalation-alerting use case (a 0.95 should read as
more urgent than a 0.6, not just both "critical").

## 4. Error analysis: most mistakes are boundary cases, not obvious failures

Sampling the false negatives and false positives shows a clear pattern:

- **False negatives** (missed critical feedback) are almost entirely the "neutral" template class
  ("Decent experience overall, app was okay.") — these were deliberately generated with urgency scores
  in the 0.35–0.6 range, straddling the 0.5 cutoff by design (see Milestone 1's ambiguous-case fix). The
  model isn't failing to understand obviously urgent text; it's landing on the wrong side of a
  genuinely ambiguous boundary the dataset was built to contain.
- **False positives** include at least one case that is actually **correct model behavior against
  corrupted ground truth**: *"System latency is terrible after the server migration, nothing works."*
  is a `CRITICAL_TEMPLATES` sentence that should score 0.65–1.0, but appears here with a "true" label
  under 0.5 — a direct hit from the ~3% label-noise injection (Milestone 1), which flips both label and
  score for a random subset of records. The model correctly read the semantic content as urgent; the
  "ground truth" for that one row was deliberately corrupted to simulate annotator disagreement. This is
  worth stating plainly: **not every model "error" against V2's labels is a model failure — some are the
  noise injection working as designed.**

## 5. Channel and aspect category carry almost no signal (a known synthetic-data limitation)

| Channel | Mean urgency | n |
|---|---|---|
| Mobile App | 0.388 | 792 |
| Shipping (App Crash aspect) | 0.390 | 664 |
| Twitter | 0.369 | 789 |
| Call Center Note | 0.375 | 805 |
| Email Support | 0.377 | 808 |
| Web Review | 0.378 | 806 |

Channel means span only **0.369–0.388** — essentially flat. The chi-squared test of channel vs. urgency
bucket independence returns **χ²=4.18, p=0.840** — nowhere near significant. Aspect category behaves the
same way (0.353–0.390 across categories).

**This is an honest limitation of the synthetic generator, not a finding about real customer behavior**:
`scripts/generate_data.py` assigns `channel` and `aspect_category` uniformly at random, independent of
the generated text and its urgency score. In a real deployment, channel almost certainly *would* carry
signal (e.g., public Twitter complaints skew toward visible outages; Call Center Notes skew toward
billing). Flagged in `docs/ENHANCEMENTS.md` as a concrete fix: correlate channel/aspect assignment with
the sampled template category during generation, so the statistical-independence test has a real
alternative hypothesis to detect instead of testing against manufactured randomness.

**Downstream implication for clustering**: this also explains why the K-Means clusters explored in
`pages/2_Clustering_Explorer.py` group records primarily by *phrasing template*, not by
`aspect_category` — the metadata field simply isn't correlated with the text, so there's no aspect signal
for TF-IDF clustering to recover even if it wanted to.

## 6. What this means for the product

- The 0.5 threshold is a reasonable default; no urgent need to recalibrate before a pilot deployment.
- Expect roughly **1 in 4 critical alerts to be a false alarm** and **roughly 1 in 5 real crises to be
  missed** at the current threshold — worth stating explicitly to any stakeholder evaluating the
  escalation-alerting use case (see `docs/PRODUCT_REVIEW.md` for the business framing).
- The urgency score's strong correlation with ground truth (0.869) means it's usable as a *ranking*
  signal (e.g., "show me the 20 most urgent open tickets") even where the binary cutoff is imperfect.
- Channel-based alerting rules ("escalate faster for Twitter") would be unsupported by this dataset and
  should not be built on top of it without first fixing the generator or validating against real data.
