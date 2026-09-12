"""HuggingFace Transformer pipelines: NER + abstractive summarization.

Both pipelines are lazy-loaded and memory-heavy (BERT-large + BART-large ≈ 2.5GB
combined), so callers should hold onto a single instance and Streamlit pages
should wrap construction in st.cache_resource rather than importing eagerly.
"""
from functools import lru_cache

NER_MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"


@lru_cache(maxsize=1)
def get_ner_pipeline():
    from transformers import pipeline
    # device=-1 pins to CPU explicitly (Streamlit Community Cloud has no GPU; this also
    # silences the pipeline's "accelerator available but no device passed" warning on
    # machines that do have one, e.g. Apple Silicon MPS during local dev).
    return pipeline("ner", model=NER_MODEL_NAME, aggregation_strategy="simple", device=-1)


@lru_cache(maxsize=1)
def get_summarizer_pipeline():
    from transformers import pipeline
    return pipeline(
        "summarization", model=SUMMARIZER_MODEL_NAME, device=-1,
        clean_up_tokenization_spaces=True,
    )


def extract_entities(text: str) -> list:
    if not text or not text.strip():
        return []
    ner = get_ner_pipeline()
    results = ner(text)
    return [
        {"entity_group": r["entity_group"], "word": r["word"], "score": float(r["score"])}
        for r in results
    ]


def summarize_batch(texts: list, max_length: int = 130, min_length: int = 30, chunk_chars: int = 3000) -> str:
    """Condenses a batch of feedback texts into an executive summary.

    Concatenates then chunks by character budget (BART's ~1024 token limit),
    summarizes each chunk, and — if more than one chunk — summarizes the
    summaries once more to keep the final brief short regardless of batch size.
    """
    if not texts:
        return ""
    summarizer = get_summarizer_pipeline()
    joined = " ".join(t for t in texts if t and t.strip())
    if not joined:
        return ""

    def _bounded_lengths(text: str) -> tuple:
        # BART's word-count-ish token estimate; keeps max/min sane for short inputs so the
        # pipeline doesn't warn about max_length exceeding input_length.
        approx_tokens = max(len(text.split()), 1)
        capped_max = min(max_length, max(approx_tokens - 2, 10))
        capped_min = min(min_length, max(capped_max - 5, 5))
        return capped_max, capped_min

    chunks = [joined[i:i + chunk_chars] for i in range(0, len(joined), chunk_chars)]
    partial_summaries = []
    for chunk in chunks:
        cmax, cmin = _bounded_lengths(chunk)
        out = summarizer(chunk, max_length=cmax, min_length=cmin, do_sample=False)
        partial_summaries.append(out[0]["summary_text"])

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined = " ".join(partial_summaries)
    cmax, cmin = _bounded_lengths(combined)
    final = summarizer(combined, max_length=cmax, min_length=cmin, do_sample=False)
    return final[0]["summary_text"]
