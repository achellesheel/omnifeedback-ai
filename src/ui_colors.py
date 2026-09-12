"""Shared risk-level -> Streamlit color mapping.

Streamlit's `:color[text]` markdown annotation only recognizes a fixed keyword
set (blue, green, orange, red, violet, gray/grey, rainbow) — "yellow" is not
among them and silently breaks the annotation instead of raising an error.
This constant is the single source of truth so the mistake can't be
reintroduced independently on one page while staying fixed on another.
"""

STREAMLIT_VALID_COLORS = {"blue", "green", "orange", "red", "violet", "gray", "grey", "rainbow", "primary"}

RISK_LEVEL_COLORS = {
    "CRITICAL": "red",
    "HIGH": "orange",
    "MEDIUM": "violet",
    "LOW": "green",
}
