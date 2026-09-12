import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.genai_copilot import generate_resolution, _parse_json_response, _risk_level_from_score


def test_local_backend_used_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = generate_resolution("App keeps crashing during checkout!", 0.9)
    assert result["_backend"] == "local_rules"
    assert result["risk_level"] == "CRITICAL"
    required = {"thought_process", "root_cause", "risk_level", "recommended_action", "customer_reply_draft"}
    assert required.issubset(result.keys())


def test_risk_level_thresholds():
    assert _risk_level_from_score(0.95) == "CRITICAL"
    assert _risk_level_from_score(0.65) == "HIGH"
    assert _risk_level_from_score(0.4) == "MEDIUM"
    assert _risk_level_from_score(0.1) == "LOW"


def test_parse_json_response_strips_markdown_fences():
    raw = '```json\n{"thought_process": "a", "root_cause": "b", "risk_level": "LOW", ' \
          '"recommended_action": "c", "customer_reply_draft": "d"}\n```'
    parsed = _parse_json_response(raw)
    assert parsed["root_cause"] == "b"


def test_parse_json_response_raises_on_missing_keys():
    import pytest
    with pytest.raises(ValueError):
        _parse_json_response('{"thought_process": "a"}')
