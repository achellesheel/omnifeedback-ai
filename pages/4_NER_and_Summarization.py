"""Batch executive summarization (BART) + Named Entity Recognition (BERT)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.app_state import select_variant
from src.database import WarehouseManager
from src.transformer_nlp import extract_entities, summarize_batch

st.set_page_config(page_title="NER & Summarization", page_icon="📝", layout="wide")
st.title("📝 Executive Briefing & Entity Extraction")
st.caption(
    "DistilBART (`sshleifer/distilbart-cnn-12-6`) condenses batches of critical feedback into an "
    "executive brief; BERT NER (`dslim/bert-base-NER`) extracts named entities. Models are ~1.7GB "
    "combined and load lazily on first use — the first click on this page will be slow."
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

# De-duplicate before slicing top-N. The synthetic generator draws critical feedback from
# only ~8 template families (see scripts/generate_data.py), so the raw top-N by urgency
# score is often mostly repeats of the same handful of sentences with different filler
# words ("Been on hold for an hour/two hours/45 minutes...") — feeding BART 3-4 near-
# duplicate copies wastes its limited context on redundant text instead of surfacing the
# real diversity of complaint types, and previously made the summary barely change between
# e.g. n=17 and n=46. Deduping first means the slider actually controls how much distinct
# content BART sees.
sorted_df = df.sort_values("urgency_score", ascending=False)
deduped = sorted_df.drop_duplicates(subset="raw_text")
selected_df = deduped.head(n)
critical_texts = selected_df["raw_text"].tolist()
n_available = len(deduped)
st.caption(
    f"{len(critical_texts)} distinct complaints selected (out of {n_available} unique critical texts "
    f"available in this dataset — duplicates with different filler words are merged before summarizing)."
)

with st.expander(f"👀 View the {len(critical_texts)} tickets behind this summary"):
    st.dataframe(
        selected_df[["feedback_id", "channel", "aspect_category", "urgency_score", "raw_text"]]
        .rename(columns={
            "feedback_id": "ID", "channel": "Channel", "aspect_category": "Aspect",
            "urgency_score": "Urgency", "raw_text": "Ticket Text",
        }),
        use_container_width=True, hide_index=True,
    )

if st.button("Generate Executive Summary", type="primary"):
    with st.spinner(f"Summarizing {len(critical_texts)} distinct critical feedback records with BART "
                     f"(first load can take ~30s)..."):
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
