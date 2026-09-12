"""Batch executive summarization (BART) + Named Entity Recognition (BERT)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.app_state import select_variant
from src.database import WarehouseManager
from src.transformer_nlp import extract_entities, summarize_batch

st.set_page_config(page_title="NER & Summarization", page_icon="📝", layout="wide")
st.title("📝 Executive Briefing & Entity Extraction")
st.caption(
    "BART (`facebook/bart-large-cnn`) condenses batches of critical feedback into an executive brief; "
    "BERT NER (`dbmdz/bert-large-cased-finetuned-conll03-english`) extracts named entities. "
    "Models are ~2.5GB combined and load lazily on first use — the first click on this page will be slow."
)

variant = select_variant()
st.info(f"Feedback source: **{variant['label']}**", icon="🔀")

db = WarehouseManager(db_path=variant["db_path"])
df = db.fetch_all_feedback()
if df.empty:
    st.warning("Warehouse is empty. Run the pipeline first.")
    st.stop()

st.subheader("Batch Executive Summary")
n = st.slider("Number of most-critical feedback records to summarize", 5, 50, 15)
critical_texts = df.sort_values("urgency_score", ascending=False)["raw_text"].head(n).tolist()

if st.button("Generate Executive Summary", type="primary"):
    with st.spinner(f"Summarizing {n} critical feedback records with BART (first load can take ~30s)..."):
        summary = summarize_batch(critical_texts)
    st.success("Executive Summary")
    st.write(summary)

st.divider()
st.subheader("Named Entity Extraction")
sample_text = st.text_area(
    "Feedback text to extract entities from",
    value="The iPhone 15 crashed right after the iOS 17.2 update, and Apple support in California hasn't responded.",
    height=100,
)
if st.button("Extract Entities"):
    with st.spinner("Running BERT NER (first load can take ~15s)..."):
        entities = extract_entities(sample_text)
    if entities:
        st.dataframe(entities, use_container_width=True, hide_index=True)
    else:
        st.info("No named entities detected in this text.")
