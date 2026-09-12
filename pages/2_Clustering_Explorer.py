"""Unsupervised aspect clustering explorer: elbow/silhouette justification + cluster contents."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from src.app_state import select_variant
from src.database import WarehouseManager
from src.ml_models import load_artifacts

st.set_page_config(page_title="Clustering Explorer", page_icon="🧩", layout="wide")
st.title("🧩 Unsupervised Aspect Clustering")
st.caption("K-Means over TF-IDF features, with the elbow/silhouette sweep used to justify the chosen K.")

variant = select_variant()
st.info(f"Viewing: **{variant['label']}**", icon="🔀")

summary_path = variant["model_dir"] / "training_summary.json"
if not summary_path.exists():
    st.error(f"No training summary found for {variant['label']}. Run `python scripts/run_pipeline.py` first.")
    st.stop()

with open(summary_path) as f:
    summary = json.load(f)

st.subheader("Why K = {}?".format(summary["best_k"]))
sweep_df = pd.DataFrame([
    {"k": int(k), "silhouette": v["silhouette"], "inertia": v["inertia"]}
    for k, v in summary["k_sweep_scores"].items()
]).sort_values("k")
col1, col2 = st.columns(2)
with col1:
    fig = px.line(sweep_df, x="k", y="silhouette", markers=True, title="Silhouette Score vs K (higher = better)")
    fig.add_vline(x=summary["best_k"], line_dash="dash", line_color="green")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = px.line(sweep_df, x="k", y="inertia", markers=True, title="Elbow Method: Inertia vs K")
    fig2.add_vline(x=summary["best_k"], line_dash="dash", line_color="green")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Cluster Contents")

db = WarehouseManager(db_path=variant["db_path"])
df = db.fetch_all_feedback()
if df.empty:
    st.warning("Warehouse is empty.")
    st.stop()

artifacts = load_artifacts(model_dir=variant["model_dir"])
X = artifacts["vectorizer"].transform(df["cleaned_text"])
df["cluster"] = artifacts["kmeans"].predict(X)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X.toarray())
df["pca_x"], df["pca_y"] = coords[:, 0], coords[:, 1]

fig3 = px.scatter(
    df, x="pca_x", y="pca_y", color=df["cluster"].astype(str),
    hover_data=["aspect_category", "channel"], title="Feedback Clusters (PCA-projected TF-IDF space)"
)
st.plotly_chart(fig3, use_container_width=True)

selected_cluster = st.selectbox("Inspect cluster", sorted(df["cluster"].unique()))
subset = df[df["cluster"] == selected_cluster]
st.write(f"**{len(subset)} records** — dominant aspect categories:")
st.bar_chart(subset["aspect_category"].value_counts())
st.dataframe(subset[["raw_text", "aspect_category", "urgency_score"]].head(15), use_container_width=True, hide_index=True)
