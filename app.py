"""OmniFeedback AI — Executive Overview (Streamlit multi-page entrypoint)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_state import select_variant
from src.database import WarehouseManager

st.set_page_config(page_title="OmniFeedback AI", page_icon="📊", layout="wide")


@st.cache_resource
def get_db(db_path):
    return WarehouseManager(db_path=db_path)


def load_feedback(db_path):
    return get_db(db_path).fetch_all_feedback()


def main():
    st.title("📊 OmniFeedback AI — Executive Overview")
    st.caption("Enterprise customer feedback analytics, escalation triage & GenAI intelligence copilot")

    variant = select_variant()
    st.info(f"Viewing: **{variant['label']}**", icon="🔀")

    df = load_feedback(variant["db_path"])
    if df.empty:
        st.warning(
            "No data for this variant yet. Run `python scripts/generate_data.py` then "
            "`python scripts/run_pipeline.py v1_naive v2_hardened` to build both variants."
        )
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_critical"] = df["urgency_score"] >= 0.5

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Feedback Records", f"{len(df):,}")
    col2.metric("Critical Rate", f"{df['is_critical'].mean() * 100:.1f}%")
    col3.metric("Avg Urgency Score", f"{df['urgency_score'].mean():.3f}")
    col4.metric("Channels Monitored", df["channel"].nunique())

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Urgency Distribution by Channel")
        fig = px.box(df, x="channel", y="urgency_score", color="channel", points=False)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Feedback Volume by Aspect Category")
        counts = df["aspect_category"].value_counts().reset_index()
        counts.columns = ["aspect_category", "count"]
        fig2 = px.bar(counts, x="aspect_category", y="count", color="aspect_category")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Urgency Trend Over Time")
    trend = df.set_index("timestamp").resample("6h")["urgency_score"].mean().reset_index()
    fig3 = px.line(trend, x="timestamp", y="urgency_score", markers=True)
    fig3.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Escalation threshold")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Recent Critical Feedback")
    critical = df[df["is_critical"]].sort_values("timestamp", ascending=False).head(10)
    st.dataframe(
        critical[["timestamp", "channel", "aspect_category", "raw_text", "urgency_score"]],
        use_container_width=True,
        hide_index=True,
    )

    st.info("Use the sidebar pages for the live inference playground, clustering explorer, "
            "GenAI copilot, and statistical tests.")


if __name__ == "__main__":
    main()
