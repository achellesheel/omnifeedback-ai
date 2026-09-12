"""Regression test for a real bug: Streamlit's `:color[text]` markdown annotation
only recognizes a fixed keyword set. Using "yellow" (not in that set) silently
breaks rendering instead of raising an error — this test fails loudly instead."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui_colors import RISK_LEVEL_COLORS, STREAMLIT_VALID_COLORS


def test_all_risk_colors_are_valid_streamlit_colors():
    for level, color in RISK_LEVEL_COLORS.items():
        assert color in STREAMLIT_VALID_COLORS, (
            f"{level} maps to '{color}', which Streamlit's :color[text] markdown "
            f"syntax does not support — it will silently fail to render."
        )


def test_all_four_risk_levels_present():
    assert set(RISK_LEVEL_COLORS.keys()) == {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
