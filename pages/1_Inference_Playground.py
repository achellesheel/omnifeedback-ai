"""Live single-text urgency scoring — works with any variant's model backend
(BiLSTM for V1/V2, DistilBERT transfer-learning for V3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.app_state import select_variant
from src.ingestor import TextIngestor
from src.ui_colors import RISK_LEVEL_COLORS
from src.urgency import load_urgency_model, score_urgency

st.set_page_config(page_title="Inference Playground", page_icon="⚡", layout="wide")
st.title("⚡ Real-Time Escalation Playground")
st.caption("Type or paste customer feedback to get an instant urgency score from the selected model.")

variant = select_variant()
st.info(f"Scoring with: **{variant['label']}**", icon="🔀")


@st.cache_resource
def get_bundle(model_dir, model_type):
    return load_urgency_model(variant)


try:
    bundle = get_bundle(variant["model_dir"], variant.get("model_type"))
except FileNotFoundError:
    st.error(f"No trained model found for {variant['label']}. Run the matching training script first "
             f"(`scripts/run_pipeline.py` for V1/V2, `scripts/train_v3.py` for V3).")
    st.stop()

text = st.text_area(
    "Customer feedback text",
    value="App keeps crashing during checkout, I'm losing customers!",
    height=100,
)

if st.button("Score Urgency", type="primary"):
    cleaned = TextIngestor.clean_text(text)
    score = score_urgency(bundle, text, cleaned_text=cleaned)

    if score >= 0.8:
        level = "CRITICAL"
    elif score >= 0.5:
        level = "HIGH"
    elif score >= 0.3:
        level = "MEDIUM"
    else:
        level = "LOW"
    color = RISK_LEVEL_COLORS[level]

    col1, col2 = st.columns(2)
    col1.metric("Urgency Score", f"{score:.3f}")
    col2.markdown(f"### Risk Level: :{color}[{level}]")
    st.progress(min(max(score, 0.0), 1.0))

    if score >= 0.5:
        st.warning("⚠️ This feedback would trigger an escalation alert in production.")
    if bundle["type"] == "bilstm":
        st.caption(f"Cleaned text fed to model: `{cleaned}`")
    else:
        st.caption("Raw text fed directly to the pretrained tokenizer (no aggressive cleaning needed).")
