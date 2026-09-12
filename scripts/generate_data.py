"""Generates both dataset variants used by the app's live "V1 vs V2" comparison:

- V1 "naive": reproduces the original capstone notebook's generator almost exactly —
  verbatim phrases picked from a tiny fixed list per class (4 negative / 4 positive),
  a flat 40% negative rate, no ambiguous cases, no label noise. Trivially separable.
- V2 "hardened": templated + randomized phrasing, deliberately ambiguous mixed-sentiment
  cases straddling the urgency threshold, ~3% simulated label noise, and a ~15% critical
  rate to mirror real-world crisis rarity. This is the version the production app ships.

Both are saved so the Streamlit app can load either one and show the metric difference
live, instead of the difference only living in PROGRESS.md prose.
"""
import os
import random
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.variants import VARIANTS  # noqa: E402

os.makedirs(os.path.dirname(VARIANTS["v1_naive"]["raw_csv"]), exist_ok=True)

CHANNELS = ["Mobile App", "Twitter", "Email Support", "Web Review", "Call Center Note"]
ASPECTS = ["Billing Defect", "App Crash", "Shipping Delay", "UI Ergonomics", "Product Quality", "General Praise"]

# --- V1 naive: verbatim phrase lists, matching the original capstone notebook ---
NAIVE_NEGATIVE_PHRASES = [
    "App keeps crashing during checkout payment step!",
    "Double charged on my credit card, fix this immediately!",
    "Package arrived damaged and customer support is ignoring my emails.",
    "System latency is terrible after the latest update.",
    "Major outage across the platform, unable to log in.",
]
NAIVE_POSITIVE_PHRASES = [
    "Love the new design update! Very smooth.",
    "Customer support resolved my billing issue in 5 minutes.",
    "Great quality product, highly recommended!",
    "Fast shipping and nice packaging.",
]


def generate_naive(num_records: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Verbatim-phrase generator: same structure/rate as the original capstone notebook."""
    rng = random.Random(seed)
    base_time = datetime.now() - timedelta(days=30)
    records = []
    for i in range(1, num_records + 1):
        is_negative = rng.random() < 0.4
        if is_negative:
            text = rng.choice(NAIVE_NEGATIVE_PHRASES)
            urgency_score = round(rng.uniform(0.6, 1.0), 4)
            label = 0
        else:
            text = rng.choice(NAIVE_POSITIVE_PHRASES)
            urgency_score = round(rng.uniform(0.0, 0.4), 4)
            label = 1
        records.append({
            "feedback_id": 10000 + i,
            "user_id": rng.randint(100, 999),
            "raw_text": text,
            "channel": rng.choice(CHANNELS),
            "aspect_category": rng.choice(ASPECTS),
            "urgency_score": urgency_score,
            "label": label,
            "timestamp": (base_time + timedelta(minutes=rng.randint(0, 43200))).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return pd.DataFrame(records)

CRITICAL_TEMPLATES = [
    "App keeps crashing during {step} step, this is unacceptable!",
    "Double charged on my {payment} for {product}, fix this immediately!",
    "Package arrived {condition} and support is completely ignoring my emails.",
    "System latency is terrible after the {event}, nothing works.",
    "Major outage across the platform, unable to {action} for hours.",
    "Your {product} deleted all my data without warning, I need this fixed NOW.",
    "Been on hold for {duration} trying to reach a human, this is ridiculous.",
    "The {feature} is completely broken since the last update, losing customers.",
]
CRITICAL_FILLS = {
    "step": ["checkout", "payment", "login", "upload"],
    "payment": ["credit card", "debit card", "PayPal account"],
    "product": ["order", "subscription", "account", "invoice"],
    "condition": ["damaged", "empty", "the wrong item"],
    "event": ["latest update", "server migration", "new release"],
    "action": ["log in", "check out", "reach support"],
    "duration": ["45 minutes", "an hour", "two hours"],
    "feature": ["search bar", "notifications", "export tool", "dashboard"],
}

POSITIVE_TEMPLATES = [
    "Love the new {feature} update! Very {adjective}.",
    "Customer support resolved my {issue} in {duration}, amazing team.",
    "Great {quality} product, highly recommended to everyone!",
    "Fast shipping and {packaging} packaging, will order again.",
    "The {feature} redesign makes everything so much {adjective}.",
    "Really impressed with how {adjective} the {product} experience is now.",
]
POSITIVE_FILLS = {
    "feature": ["design", "dashboard", "checkout flow", "search"],
    "adjective": ["smooth", "intuitive", "fast", "easy"],
    "issue": ["billing issue", "login problem", "shipping question"],
    "duration": ["5 minutes", "under an hour", "one message"],
    "quality": ["excellent", "reliable", "well-built"],
    "packaging": ["nice", "eco-friendly", "secure"],
    "product": ["onboarding", "checkout", "mobile app"],
}

NEUTRAL_TEMPLATES = [
    "The {feature} works fine but could use minor improvements.",
    "Decent experience overall, {feature} was okay.",
    "It does what it says, nothing special about the {feature}.",
]
NEUTRAL_FILLS = {"feature": ["app", "website", "checkout", "support chat"]}

# Ambiguous/borderline cases: mix positive and negative vocabulary in one message so
# classifiers can't rely purely on keyword overlap — mirrors real mixed-sentiment reviews.
AMBIGUOUS_TEMPLATES = [
    "The {feature} update looks great but it {problem} for me, mixed feelings.",
    "Support was {tone} about my {issue}, though it took a while to {action}.",
    "Mostly happy with the {product}, just wish the {feature} didn't {problem}.",
    "Not terrible, but the {feature} still {problem} sometimes after the fix.",
]
AMBIGUOUS_FILLS = {
    "feature": ["checkout", "search", "notifications", "dashboard"],
    "problem": ["lags a bit", "crashed once", "logged me out", "loaded slowly"],
    "tone": ["polite", "helpful", "friendly"],
    "issue": ["refund request", "login problem", "shipping question"],
    "action": ["get resolved", "hear back", "get a straight answer"],
    "product": ["app", "service", "subscription"],
}


def _fill(template: str, fills: dict) -> str:
    for key, options in fills.items():
        if f"{{{key}}}" in template:
            template = template.replace(f"{{{key}}}", random.choice(options), 1)
    return template


def generate_hardened(
    num_records: int = 4000, critical_rate: float = 0.15, neutral_rate: float = 0.20,
    ambiguous_rate: float = 0.15, label_noise_rate: float = 0.03, seed: int = 42,
) -> pd.DataFrame:
    rng_state = random.getstate()
    random.seed(seed)
    base_time = datetime.now() - timedelta(days=30)
    records = []
    for i in range(1, num_records + 1):
        roll = random.random()
        if roll < critical_rate:
            text = _fill(random.choice(CRITICAL_TEMPLATES), CRITICAL_FILLS)
            urgency_score = round(random.uniform(0.65, 1.0), 4)
            label = 0
        elif roll < critical_rate + neutral_rate:
            text = _fill(random.choice(NEUTRAL_TEMPLATES), NEUTRAL_FILLS)
            urgency_score = round(random.uniform(0.35, 0.6), 4)
            label = 1
        elif roll < critical_rate + neutral_rate + ambiguous_rate:
            text = _fill(random.choice(AMBIGUOUS_TEMPLATES), AMBIGUOUS_FILLS)
            # ambiguous cases straddle the escalation threshold on purpose
            urgency_score = round(random.uniform(0.4, 0.7), 4)
            label = 0 if urgency_score >= 0.5 else 1
        else:
            text = _fill(random.choice(POSITIVE_TEMPLATES), POSITIVE_FILLS)
            urgency_score = round(random.uniform(0.0, 0.3), 4)
            label = 1

        # Small fraction of realistic label noise (mislabeled/annotator disagreement),
        # so downstream models can't trivially hit a perfect score on this dataset.
        if random.random() < label_noise_rate:
            label = 1 - label
            urgency_score = round(1.0 - urgency_score, 4)

        records.append({
            "feedback_id": 10000 + i,
            "user_id": random.randint(100, 999),
            "raw_text": text,
            "channel": random.choice(CHANNELS),
            "aspect_category": random.choice(ASPECTS),
            "urgency_score": urgency_score,
            "label": label,
            "timestamp": (base_time + timedelta(minutes=random.randint(0, 43200))).strftime("%Y-%m-%d %H:%M:%S"),
        })

    random.setstate(rng_state)
    return pd.DataFrame(records)


if __name__ == "__main__":
    for variant_id, generator in [("v1_naive", generate_naive), ("v2_hardened", generate_hardened)]:
        df = generator()
        out_path = VARIANTS[variant_id]["raw_csv"]
        df.to_csv(out_path, index=False)
        print(f"[{variant_id}] Generated {len(df)} rows -> {out_path}")
        print(df["label"].value_counts(normalize=True))
        print()
