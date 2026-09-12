"""GenAI Resolution Copilot: CoT + few-shot prompting with structured JSON output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.app_state import select_variant
from src.genai_copilot import generate_resolution
from src.ingestor import TextIngestor
from src.ui_colors import RISK_LEVEL_COLORS
from src.urgency import load_urgency_model, score_urgency

st.set_page_config(page_title="GenAI Copilot", page_icon="🤖", layout="wide")
st.title("🤖 GenAI Resolution Copilot")
st.caption(
    "Chain-of-Thought + few-shot prompting drafts a structured triage response. "
    "Uses Claude if ANTHROPIC_API_KEY is configured in secrets, otherwise a deterministic local backend."
)

variant = select_variant()
st.info(f"Urgency scored with: **{variant['label']}**", icon="🔀")


@st.cache_resource
def get_bundle(model_dir, model_type):
    return load_urgency_model(variant)


text = st.text_area(
    "Customer feedback",
    value="Double charged on my credit card for the same order, and support hasn't replied in 2 days!",
    height=100,
)

if st.button("Generate Resolution", type="primary"):
    try:
        bundle = get_bundle(variant["model_dir"], variant.get("model_type"))
        cleaned = TextIngestor.clean_text(text)
        urgency_score = score_urgency(bundle, text, cleaned_text=cleaned)
    except FileNotFoundError:
        urgency_score = 0.5
        st.info("No trained model found for this variant — using a neutral default urgency score of 0.5.")

    with st.spinner("Reasoning through root cause..."):
        result = generate_resolution(text, urgency_score)

    backend_label = "🔮 Claude API" if result["_backend"] == "claude" else "⚙️ Local rule-based (no API key configured)"
    st.caption(f"Backend used: {backend_label} · Urgency score: {urgency_score:.3f}")

    st.markdown(f"### Risk Level: :{RISK_LEVEL_COLORS.get(result['risk_level'], 'blue')}[{result['risk_level']}]")

    with st.expander("💭 Chain-of-Thought reasoning", expanded=True):
        st.write(result["thought_process"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Root Cause")
        st.write(result["root_cause"])
        st.subheader("Recommended Action")
        st.write(result["recommended_action"])
    with col2:
        st.subheader("Customer Reply Draft")
        st.text_area("", value=result["customer_reply_draft"], height=150, disabled=True, label_visibility="collapsed")

    st.json(result)
