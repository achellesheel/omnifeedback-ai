import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import WarehouseManager


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        yield WarehouseManager(db_path=Path(tmp) / "test.db")


def _sample_df(n=3):
    return pd.DataFrame({
        "feedback_id": range(1, n + 1),
        "user_id": [101] * n,
        "raw_text": ["raw text"] * n,
        "cleaned_text": ["clean text"] * n,
        "channel": ["Twitter"] * n,
        "aspect_category": ["App Crash"] * n,
        "urgency_score": [0.5] * n,
        "label": [1] * n,
        "timestamp": ["2026-01-01 00:00:00"] * n,
    })


def test_schema_created(db):
    tables = db.query("SELECT name FROM sqlite_master WHERE type='table'")["name"].tolist()
    for expected in ["Fact_Feedback", "Dim_User", "Dim_Channel", "Dim_Product"]:
        assert expected in tables


def test_insert_batch_roundtrip(db):
    inserted = db.insert_feedback_batch(_sample_df(3))
    assert inserted == 3
    result = db.fetch_all_feedback()
    assert len(result) == 3
    assert set(result["channel"]) == {"Twitter"}


def test_rollback_on_bad_urgency_score(db):
    df = _sample_df(2)
    df.loc[1, "urgency_score"] = 5.0  # violates CHECK constraint (0.0-1.0)
    with pytest.raises(Exception):
        db.insert_feedback_batch(df)
    result = db.fetch_all_feedback()
    assert len(result) == 0  # entire batch rolled back, nothing partially committed


def test_missing_required_column_raises(db):
    df = _sample_df(1).drop(columns=["urgency_score"])
    with pytest.raises(ValueError):
        db.insert_feedback_batch(df)


def test_dim_tables_deduplicate(db):
    db.insert_feedback_batch(_sample_df(5))
    channels = db.query("SELECT * FROM Dim_Channel")
    assert len(channels) == 1  # same channel across all 5 rows, not duplicated
