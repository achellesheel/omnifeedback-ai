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

## 6. V3: the same limitation, measured — and where it draws the line

Milestone 8 (`PROGRESS.md`) found that V2's from-scratch, 196-word BiLSTM vocabulary collapsed real-world
critical and positive feedback to nearly identical scores (0.02 separation). V3 (a transfer-learned
DistilBERT, frozen early layers, ~21% of parameters trainable) fixed this — 0.30 separation on the same
sentences, ~13x larger, with no more training data than V2 had.

**But V3's CRITICAL threshold (≥0.8) is narrower than "how bad this sounds."** Testing ~20 real-world
severity statements directly against the model found that genuinely severe complaints ("production API
down for 45 minutes, losing revenue every minute") reliably reach HIGH (0.56–0.79) but not CRITICAL.
Only two phrasing patterns reliably cross 0.8:

| Pattern | Example | Score |
|---|---|---|
| Data deletion + demand for immediate fix + "unacceptable" | "Your platform deleted all of my account data without any warning, I need this fixed immediately, this is completely unacceptable." | 0.860 |
| Unauthorized account access + password change + financial data at risk | "Someone accessed my account without my permission and changed my password, I am now completely locked out and my financial data is at serious risk." | 0.808 |

Both map closely to a specific template family in V2's hardened training data ("Your {product} deleted
all my data without warning, I need this fixed NOW."). This is expected behavior for a model calibrated
to its training distribution, not a bug — but it means the 0.8 cutoff should be read as "matches the
data-loss/account-compromise pattern strongly," not "the model agrees this is maximally severe." Training
on genuinely diverse real complaint text would likely broaden what reaches CRITICAL (see §7 — validated,
not yet trained on).

## 7. Real-data validation: does the V3 fix replicate outside the lab?

Section 6's 0.30 separation came from 8 sentences written by hand — a fair challenge is whether that's
"easy mode." `scripts/validate_real_data.py` reruns the same measurement on real, independently-authored
text: 150 Yelp 1-star vs. 150 5-star reviews (proxy-critical/proxy-low), and the same split on 150+150
Sentiment140 tweets.

| | Yelp (n=150/class) | Sentiment140 (n=150/class) |
|---|---|---|
| V2 separation | 0.0249 (p=0.0048 — significant, tiny) | 0.0050 (p=0.39 — **not significant**) |
| V3 separation | 0.1272 (p<0.0001) | 0.0837 (p<0.0001) |
| V2 vs. full 5-star scale | Spearman ρ=0.073 (p=0.14, not significant) | — |
| V3 vs. full 5-star scale | Spearman ρ=0.682 (p<0.0001) | — |

**The core finding replicates**: V3's separation is real and statistically significant on independent
data the model never trained on, on both a review dataset and a social-media dataset. **The honest
correction**: the effect size (0.08–0.13) is meaningfully smaller than the hand-written stress test's
0.30 — real text is noisier than examples written to be unambiguous, and reporting the smaller, real
number matters more than repeating the bigger, curated one.

**A genuinely useful secondary finding**: V2 isn't uniformly blind — its separation is small but
statistically real on Yelp's longer reviews (p=0.0048), and statistically indistinguishable from noise on
Sentiment140's short tweets (p=0.39). This matches the §6 root cause exactly: shorter text leaves a
196-word vocabulary proportionally less to work with, so the failure is worse on tweets than full reviews.
V3's Spearman correlation against Yelp's complete 5-star scale (ρ=0.682) versus V2's (ρ=0.073, not
significant) is the single strongest piece of evidence in this project that V3 tracks real sentiment.

## 8. What this means for the product

- The 0.5 threshold is a reasonable default; no urgent need to recalibrate before a pilot deployment.
- Expect roughly **1 in 4 critical alerts to be a false alarm** and **roughly 1 in 5 real crises to be
  missed** at the current threshold — worth stating explicitly to any stakeholder evaluating the
  escalation-alerting use case (see `docs/PRODUCT_REVIEW.md` for the business framing).
- The urgency score's strong correlation with ground truth (0.869) means it's usable as a *ranking*
  signal (e.g., "show me the 20 most urgent open tickets") even where the binary cutoff is imperfect.
- Channel-based alerting rules ("escalate faster for Twitter") would be unsupported by this dataset and
  should not be built on top of it without first fixing the generator or validating against real data.
