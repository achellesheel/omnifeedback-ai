"""Fetches real-world review/sentiment samples for later comparison against the synthetic
dataset (see PROGRESS.md Milestone 1 for why the synthetic generator alone isn't sufficient
evidence of a good model). Not part of the main pipeline — a research/insights-phase input.

Fixes the original capstone notebook's dead sources:
  - stanfordnlp/sentiment140 and sentiment140 both fail ("Dataset scripts are no longer
    supported"); contemmcm/sentiment140 is a maintained Parquet mirror with a `complete`
    split (not `train`) and the same {text, label} schema.
  - Yelp/yelp_review_full already worked in the original notebook and is unchanged here.
"""
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_yelp(n: int = 5000) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("Yelp/yelp_review_full", split=f"train[:{n}]")
    return pd.DataFrame(ds)


def fetch_sentiment140(n: int = 5000) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("contemmcm/sentiment140", split=f"complete[:{n}]")
    return pd.DataFrame(ds)


def main():
    try:
        df_yelp = fetch_yelp()
        path = os.path.join(DATA_DIR, "yelp_reviews_sample.csv")
        df_yelp.to_csv(path, index=False)
        print(f"Saved {len(df_yelp)} Yelp rows -> {path}")
    except Exception as e:
        print(f"Yelp fetch failed ({e}); skipping.")

    try:
        df_s140 = fetch_sentiment140()
        path = os.path.join(DATA_DIR, "sentiment140_sample.csv")
        df_s140.to_csv(path, index=False)
        print(f"Saved {len(df_s140)} Sentiment140 rows -> {path}")
    except Exception as e:
        print(f"Sentiment140 fetch failed ({e}); skipping.")


if __name__ == "__main__":
    main()
