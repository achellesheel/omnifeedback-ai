"""Unit tests for the model-agnostic urgency dispatcher — verifies routing logic
without downloading/loading real model weights."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.urgency import score_urgency


def test_bilstm_bundle_uses_cleaned_text():
    fake_predict = MagicMock(return_value=0.42)
    bundle = {"type": "bilstm", "model": "m", "vocab": "v", "device": "cpu"}
    import src.urgency as urgency_mod
    original = urgency_mod.predict_urgency
    urgency_mod.predict_urgency = fake_predict
    try:
        score = score_urgency(bundle, "raw TEXT!!", cleaned_text="cleaned text")
        assert score == 0.42
        fake_predict.assert_called_once_with("m", "v", "cleaned text", "cpu")
    finally:
        urgency_mod.predict_urgency = original


def test_transformer_bundle_uses_raw_text():
    fake_predict = MagicMock(return_value=0.77)
    bundle = {"type": "transformer", "model": "m", "tokenizer": "tok", "device": "cpu", "max_len": 64}
    import src.urgency as urgency_mod
    original = urgency_mod.predict_urgency_transformer
    urgency_mod.predict_urgency_transformer = fake_predict
    try:
        score = score_urgency(bundle, "raw TEXT!!", cleaned_text="cleaned text")
        assert score == 0.77
        fake_predict.assert_called_once_with("m", "tok", "raw TEXT!!", "cpu", 64)
    finally:
        urgency_mod.predict_urgency_transformer = original
