import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ml_models


@pytest.fixture(autouse=True)
def isolate_models_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ml_models, "MODELS_DIR", tmp_path)


def _sample_df(n=60):
    rng = np.random.RandomState(0)
    critical = ["app crash checkout payment error"] * (n // 3)
    positive = ["love the fast shipping great support"] * (n // 3)
    neutral = ["okay experience nothing special today"] * (n - 2 * (n // 3))
    texts = critical + positive + neutral
    labels = [0] * len(critical) + [1] * len(positive) + [1] * len(neutral)
    return pd.DataFrame({"cleaned_text": texts, "label": labels})


def test_fit_tfidf_produces_vocab():
    vec = ml_models.fit_tfidf(["hello world", "world of code", "hello code world"], max_features=10)
    assert len(vec.vocabulary_) > 0


def test_kmeans_pipeline_selects_valid_k_and_saves(tmp_path):
    df = _sample_df()
    results = ml_models.run_ml_pipeline(df, save=True)
    assert results["best_k"] in range(2, 9)
    assert (tmp_path / "tfidf.pkl").exists()
    assert (tmp_path / "kmeans_model.pkl").exists()
    assert (tmp_path / "baseline_classifier.pkl").exists()


def test_baseline_classifiers_beat_random_guessing():
    df = _sample_df()
    results = ml_models.run_ml_pipeline(df, save=False)
    for name, res in results["baseline_results"].items():
        assert res["f1"] > 0.3  # clearly separable synthetic classes, should not be near-random


def test_select_best_k_rejects_naive_argmax_at_boundary():
    # near-linear, decelerating gains that never truly plateau -> naive argmax would pick k=6 (the edge)
    k_scores = {2: {"silhouette": 0.10}, 3: {"silhouette": 0.11}, 4: {"silhouette": 0.12},
                5: {"silhouette": 0.125}, 6: {"silhouette": 0.128}}
    assert ml_models.select_best_k(k_scores) != 6


def test_select_best_k_finds_real_elbow():
    k_scores = {2: {"silhouette": 0.10}, 3: {"silhouette": 0.30}, 4: {"silhouette": 0.32}, 5: {"silhouette": 0.33}}
    assert ml_models.select_best_k(k_scores) == 3
