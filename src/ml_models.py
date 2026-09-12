"""Classical ML: TF-IDF features, K-Means aspect clustering, baseline classifiers."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, silhouette_score
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def fit_tfidf(texts: list, max_features: int = 2000) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    vectorizer.fit(texts)
    return vectorizer


def select_k_via_silhouette(X, k_range=range(2, 13), random_state: int = 42) -> dict:
    """Elbow/silhouette sweep to justify the chosen K instead of guessing."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = {
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, labels)) if k > 1 else 0.0,
        }
    return scores


def select_best_k(k_scores: dict) -> int:
    """Picks k via the Kneedle max-distance-from-chord elbow method (Satopaa et al.,
    2011), instead of naively argmax-ing silhouette.

    Template-generated text tends to form many small, ultra-tight paraphrase clusters,
    so raw silhouette can climb close to monotonically as k grows toward the number of
    templates — argmax would then just pick the largest k tested every time, which is a
    metric artifact, not a meaningful business segmentation. A fixed-threshold "stop when
    the gain gets small" rule turned out too fragile (this curve's gains wobble non-
    monotonically), so instead we find the point of maximum curvature relative to the
    straight line joining the first and last (k, silhouette) points sampled — the same
    idea as the classic elbow-in-inertia method, applied to silhouette here since higher
    silhouette is "better" (unlike inertia, which is "lower is better").
    """
    ks = sorted(k_scores.keys())
    scores = np.array([k_scores[k]["silhouette"] for k in ks], dtype=float)
    xs = np.array(ks, dtype=float)

    xs_norm = (xs - xs.min()) / (xs.max() - xs.min())
    ys_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    x1, y1 = xs_norm[0], ys_norm[0]
    x2, y2 = xs_norm[-1], ys_norm[-1]
    line_vec = np.array([x2 - x1, y2 - y1])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return ks[0]

    distances = []
    for x, y in zip(xs_norm, ys_norm):
        point_vec = np.array([x - x1, y - y1])
        cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
        distances.append(abs(cross) / line_len)

    return ks[int(np.argmax(distances))]


def fit_kmeans(X, n_clusters: int, random_state: int = 42) -> KMeans:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(X)
    return km


def train_baseline_classifiers(X_train, y_train, X_test, y_test) -> dict:
    """Train LogReg + RandomForest baselines and return metrics + fitted models."""
    results = {}
    for name, clf in [
        ("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("random_forest", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42)),
    ]:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name] = {
            "model": clf,
            "f1": float(f1_score(y_test, preds, average="macro")),
            "report": classification_report(y_test, preds, output_dict=True),
        }
    return results


def run_ml_pipeline(
    df: pd.DataFrame, text_col: str = "cleaned_text", label_col: str = "label",
    save: bool = True, model_dir: Path = None,
) -> dict:
    model_dir = model_dir or MODELS_DIR
    """End-to-end classical ML pipeline: TF-IDF -> K sweep -> KMeans -> baseline classifiers."""
    texts = df[text_col].tolist()
    vectorizer = fit_tfidf(texts)
    X = vectorizer.transform(texts)

    k_scores = select_k_via_silhouette(X)
    best_k = select_best_k(k_scores)
    kmeans = fit_kmeans(X, n_clusters=best_k)
    clusters = kmeans.predict(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )
    baseline_results = train_baseline_classifiers(X_train, y_train, X_test, y_test)

    if save:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, model_dir / "tfidf.pkl")
        joblib.dump(kmeans, model_dir / "kmeans_model.pkl")
        best_baseline_name = max(baseline_results, key=lambda n: baseline_results[n]["f1"])
        joblib.dump(baseline_results[best_baseline_name]["model"], model_dir / "baseline_classifier.pkl")

    return {
        "vectorizer": vectorizer,
        "kmeans": kmeans,
        "best_k": best_k,
        "k_sweep_scores": k_scores,
        "clusters": clusters,
        "baseline_results": {n: {"f1": r["f1"], "report": r["report"]} for n, r in baseline_results.items()},
    }


def load_artifacts(model_dir: Path = None) -> dict:
    model_dir = model_dir or MODELS_DIR
    return {
        "vectorizer": joblib.load(model_dir / "tfidf.pkl"),
        "kmeans": joblib.load(model_dir / "kmeans_model.pkl"),
        "baseline_classifier": joblib.load(model_dir / "baseline_classifier.pkl"),
    }
