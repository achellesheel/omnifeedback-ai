"""Live single-text urgency scoring with the trained BiLSTM model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.app_state import select_variant
from src.dl_lstm import load_bilstm_model, predict_urgency
from src.ingestor import TextIngestor

st.set_page_config(page_title="Inference Playground", page_icon="⚡", layout="wide")
st.title("⚡ Real-Time Escalation Playground")
st.caption("Type or paste customer feedback to get an instant urgency score from the BiLSTM regressor.")

variant = select_variant()
st.info(f"Scoring with: **{variant['label']}**", icon="🔀")


@st.cache_resource
def get_model(model_dir):
    return load_bilstm_model(model_dir=model_dir)


try:
    model, vocab, device = get_model(variant["model_dir"])
except FileNotFoundError:
    st.error(f"No trained BiLSTM model found for {variant['label']}. Run "
             f"`python scripts/run_pipeline.py` first.")
    st.stop()

text = st.text_area(
    "Customer feedback text",
    value="App keeps crashing during checkout, I'm losing customers!",
    height=100,
)

if st.button("Score Urgency", type="primary"):
    cleaned = TextIngestor.clean_text(text)
    score = predict_urgency(model, vocab, cleaned, device)

    if score >= 0.8:
        level, color = "CRITICAL", "red"
    elif score >= 0.5:
        level, color = "HIGH", "orange"
    elif score >= 0.3:
        level, color = "MEDIUM", "yellow"
    else:
        level, color = "LOW", "green"

    col1, col2 = st.columns(2)
    col1.metric("Urgency Score", f"{score:.3f}")
    col2.markdown(f"### Risk Level: :{color}[{level}]")
    st.progress(min(max(score, 0.0), 1.0))

    if score >= 0.5:
        st.warning("⚠️ This feedback would trigger an escalation alert in production.")
    st.caption(f"Cleaned text fed to model: `{cleaned}`")
