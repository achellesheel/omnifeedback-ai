"""V1 vs V2 vs V3 side-by-side comparison — both fixes, made visible."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.variants import VARIANTS

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ V1 → V2 → V3: Two Real Bugs, Found and Fixed in Public")
st.caption(
    "V1 → V2 fixed an overfitting-prone dataset (identical code, different training data). "
    "V2 → V3 fixed a generalization failure by swapping a from-scratch BiLSTM for a transfer-learned "
    "DistilBERT. Both fixes are demonstrated below with real numbers, not just claimed in prose."
)

summaries = {}
for vid, v in VARIANTS.items():
    path = v["model_dir"] / "training_summary.json"
    if path.exists():
        with open(path) as f:
            summaries[vid] = json.load(f)

if len(summaries) < 2:
    st.error(
        "Both variants need to be trained first. Run:\n\n"
        "`python scripts/generate_data.py`\n\n"
        "`python scripts/run_pipeline.py v1_naive v2_hardened`"
    )
    st.stop()

v1, v2 = summaries["v1_naive"], summaries["v2_hardened"]

st.subheader("Headline Metrics")
metric_rows = [
    ("K-Means best k (Kneedle elbow)", v1["best_k"], v2["best_k"], None),
    ("Baseline classifier macro-F1", v1["baseline_f1"]["logistic_regression"], v2["baseline_f1"]["logistic_regression"], ".3f"),
    ("BiLSTM test MSE", v1["bilstm_test_mse"], v2["bilstm_test_mse"], ".5f"),
    ("BiLSTM test macro-F1", v1["bilstm_test_f1"], v2["bilstm_test_f1"], ".3f"),
]

cols = st.columns(len(metric_rows))
for col, (label, v1_val, v2_val, fmt) in zip(cols, metric_rows):
    with col:
        st.markdown(f"**{label}**")
        v1_str = format(v1_val, fmt) if fmt else str(v1_val)
        v2_str = format(v2_val, fmt) if fmt else str(v2_val)
        delta = None
        try:
            delta = round(float(v2_val) - float(v1_val), 4)
        except (TypeError, ValueError):
            pass
        st.metric("V1 Naive", v1_str)
        st.metric("V2 Hardened", v2_str, delta=delta, delta_color="off")

st.warning(
    "⚠️ **V1's near-perfect scores are the red flag, not the win.** A silhouette of 1.0 and macro-F1 "
    "of 1.0 mean the model memorized a handful of verbatim training phrases — not that it learned to "
    "generalize. V2's lower-but-honest scores reflect data with real ambiguity, exactly like production "
    "traffic would have. See the K-sweep chart below for why this shows up mechanically, not just in "
    "the final number."
)

st.divider()
st.subheader("Why V1 is misleading: the silhouette curve never plateaus")
fig = go.Figure()
for vid, label, color in [("v1_naive", "V1 Naive", "crimson"), ("v2_hardened", "V2 Hardened", "seagreen")]:
    sweep = summaries[vid]["k_sweep_scores"]
    ks = sorted(int(k) for k in sweep.keys())
    fig.add_trace(go.Scatter(
        x=ks, y=[sweep[str(k)]["silhouette"] for k in ks],
        mode="lines+markers", name=label, line=dict(color=color),
    ))
fig.update_layout(
    xaxis_title="k (number of clusters)", yaxis_title="Silhouette score",
    title="V1 hits silhouette = 1.0 (perfect, meaningless) because only ~9 unique text strings exist "
          "in its training data; V2's curve reflects genuine, gradually-diminishing cluster separation.",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("BiLSTM Training Curves (Early Stopping)")
fig2 = go.Figure()
for vid, label, color in [("v1_naive", "V1 Naive", "crimson"), ("v2_hardened", "V2 Hardened", "seagreen")]:
    history = summaries[vid]["bilstm_history"]
    epochs = list(range(1, len(history["val_loss"]) + 1))
    fig2.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers",
                               name=f"{label} — val MSE", line=dict(color=color)))
    fig2.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines+markers",
                               name=f"{label} — train MSE", line=dict(color=color, dash="dot")))
fig2.update_layout(xaxis_title="Epoch", yaxis_title="MSE Loss")
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Sample Training Text: See the Difference Yourself")
col1, col2 = st.columns(2)
for col, vid in [(col1, "v1_naive"), (col2, "v2_hardened")]:
    with col:
        st.markdown(f"**{VARIANTS[vid]['label']}**")
        try:
            df = pd.read_csv(VARIANTS[vid]["raw_csv"])
            st.write(f"Unique raw texts: **{df['raw_text'].nunique()}** out of {len(df)} rows")
            st.dataframe(df[["raw_text", "urgency_score", "label"]].sample(5, random_state=1),
                         use_container_width=True, hide_index=True)
        except FileNotFoundError:
            st.info("Run scripts/generate_data.py first.")

st.divider()
st.header("⚡ V2 (BiLSTM) vs. V3 (Transfer-Learned DistilBERT): Generalization to Real Text")
st.caption(
    "V2's BiLSTM learns its own ~196-word vocabulary from scratch — anything outside it collapses to "
    "near-meaningless <UNK> tokens. V3 fine-tunes a pretrained DistilBERT (frozen early layers, only the "
    "last 2 layers + head trained — ~21% of parameters) so it already understands English before ever "
    "seeing this dataset. These sentences are deliberately NOT phrased like the training templates."
)

stress_path = Path(__file__).resolve().parent.parent / "models" / "stress_test_comparison.json"
if not stress_path.exists():
    st.info("Run `python scripts/stress_test_compare.py` to generate this comparison.")
else:
    with open(stress_path) as f:
        stress = json.load(f)

    gap_col1, gap_col2 = st.columns(2)
    gap_col1.metric("V2 critical-vs-low separation", stress["v2_critical_minus_low_gap"])
    gap_col2.metric("V3 critical-vs-low separation", stress["v3_critical_minus_low_gap"],
                     delta=round(stress["v3_critical_minus_low_gap"] - stress["v2_critical_minus_low_gap"], 4))

    stress_df = pd.DataFrame(stress["stress_test_results"])
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=stress_df.index, y=stress_df["v2_score"], name="V2 (BiLSTM)", marker_color="crimson"))
    fig3.add_trace(go.Bar(x=stress_df.index, y=stress_df["v3_score"], name="V3 (DistilBERT)", marker_color="seagreen"))
    fig3.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="0.5 threshold")
    fig3.update_layout(
        barmode="group", xaxis_title="Stress-test sentence (see table below)", yaxis_title="Urgency score",
        title="V2 clusters everything near 0.4-0.5 regardless of true severity. V3 separates them.",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        stress_df.rename(columns={"expected": "Expected", "text": "Text", "v2_score": "V2 Score", "v3_score": "V3 Score"}),
        use_container_width=True, hide_index=True,
    )
    st.warning(
        "⚠️ Every sentence above is real-world phrasing, not template-matching text. V2 scores CRITICAL "
        "and LOW examples almost identically (~0.02 apart) — it cannot tell them apart. V3 separates them "
        "by ~0.30, correctly ranking severity even on phrasing it never saw verbatim in training."
    )

st.divider()
with st.expander("📖 The full story: what we changed and why (from PROGRESS.md)"):
    st.markdown(
        """
1. **First run looked perfect and that was the problem.** Verbatim phrase lists made every class
   linearly separable by exact keyword overlap — 9 unique sentences in the whole dataset.
2. **We rewrote the generator**: template + randomized fill-words instead of verbatim phrases, added
   deliberately ambiguous mixed-sentiment cases straddling the urgency threshold, and injected ~3%
   label noise to simulate real annotator disagreement.
3. **We also fixed how K is chosen.** Naively maximizing silhouette score picks whatever k is largest —
   on templated text, silhouette keeps climbing as k approaches the number of underlying sentence
   templates. We replaced `argmax(silhouette)` with the **Kneedle max-distance-from-chord elbow method**,
   which is robust to this artifact.
4. **Then a real person testing the live app found a second, bigger bug**: feeding it genuinely
   natural phrasing (not matching the training templates) made V2's BiLSTM score critical and positive
   feedback almost identically — a 40-60% out-of-vocabulary rate on real text collapsed its predictions
   toward the training mean regardless of actual severity.
5. **We fixed it with transfer learning**: V3 fine-tunes a pretrained DistilBERT (frozen early layers,
   ~21% of parameters trainable) instead of learning an embedding table from scratch. Subword
   tokenization means there's no closed-vocabulary problem at all — it separates real critical and
   positive feedback by ~0.30 where V2 managed ~0.02.

Full write-up with reasoning: see `PROGRESS.md` in the repository root.
        """
    )
