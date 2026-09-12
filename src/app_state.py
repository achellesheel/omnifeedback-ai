"""Shared sidebar variant selector so every Streamlit page toggles the same V1/V2 choice.

st.session_state persists across pages within one browser session in Streamlit's
multi-page app model, so selecting a variant on any page carries over to the rest.
"""
import streamlit as st

from src.variants import DEFAULT_VARIANT, VARIANTS


def select_variant() -> dict:
    if "variant_id" not in st.session_state:
        st.session_state["variant_id"] = DEFAULT_VARIANT

    with st.sidebar:
        st.markdown("### 🔀 Dataset / Model Version")
        variant_id = st.radio(
            "Compare naive vs. hardened training",
            options=list(VARIANTS.keys()),
            format_func=lambda v: VARIANTS[v]["short_label"],
            index=list(VARIANTS.keys()).index(st.session_state["variant_id"]),
            key="variant_id",
        )
        st.caption(VARIANTS[variant_id]["description"])
        st.divider()

    return VARIANTS[variant_id]
