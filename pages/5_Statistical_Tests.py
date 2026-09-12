"""Inferential statistics: Chi-Squared test of independence between channel and urgency level."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency

from src.app_state import select_variant
from src.database import WarehouseManager

st.set_page_config(page_title="Statistical Tests", page_icon="📐", layout="wide")
st.title("📐 Inferential Statistics: Channel vs. Urgency Independence")
st.caption("Chi-Squared Test of Independence: is the distribution of urgency levels the same across channels?")

variant = select_variant()
st.info(f"Viewing: **{variant['label']}**", icon="🔀")

db = WarehouseManager(db_path=variant["db_path"])
df = db.fetch_all_feedback()
if df.empty:
    st.warning("Warehouse is empty. Run the pipeline first.")
    st.stop()


def bucket(score):
    if score >= 0.65:
        return "Critical"
    if score >= 0.35:
        return "Medium"
    return "Low"


df["urgency_bucket"] = df["urgency_score"].apply(bucket)
contingency = pd.crosstab(df["channel"], df["urgency_bucket"])

st.subheader("Contingency Table")
st.dataframe(contingency, use_container_width=True)

chi2, p_value, dof, expected = chi2_contingency(contingency)

n = contingency.to_numpy().sum()
min_dim = min(contingency.shape) - 1
cramers_v = (chi2 / (n * min_dim)) ** 0.5 if min_dim > 0 else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Chi-Squared Statistic", f"{chi2:.3f}")
col2.metric("p-value", f"{p_value:.5f}")
col3.metric("Cramér's V (effect size)", f"{cramers_v:.3f}")

alpha = 0.05
if p_value < alpha:
    st.success(
        f"p-value ({p_value:.5f}) < {alpha}: reject the null hypothesis. Urgency-level distribution "
        f"**does statistically depend** on the feedback channel (Cramér's V = {cramers_v:.3f})."
    )
else:
    st.info(
        f"p-value ({p_value:.5f}) ≥ {alpha}: fail to reject the null hypothesis. No statistically "
        f"significant association detected between channel and urgency level in this dataset."
    )

st.subheader("Urgency Bucket Proportions by Channel")
props = pd.crosstab(df["channel"], df["urgency_bucket"], normalize="index").reset_index()
props_melted = props.melt(id_vars="channel", var_name="urgency_bucket", value_name="proportion")
fig = px.bar(props_melted, x="channel", y="proportion", color="urgency_bucket", barmode="stack")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Degrees of freedom & expected frequencies"):
    st.write(f"Degrees of freedom: {dof}")
    st.dataframe(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns))
