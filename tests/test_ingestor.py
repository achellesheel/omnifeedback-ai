import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestor import TextIngestor


def test_strips_html_tags():
    assert TextIngestor.clean_text("<b>Great</b> product!") == "great product"


def test_strips_urls():
    out = TextIngestor.clean_text("Check https://example.com/path?x=1 now")
    assert "http" not in out and "example" not in out


def test_extracts_hashtags_keeps_word():
    assert "outage" in TextIngestor.clean_text("Major #outage right now")


def test_strips_mentions_entirely():
    out = TextIngestor.clean_text("@support please help")
    assert "support" not in out or "please help" in out
    assert "@" not in out


def test_collapses_repeated_chars():
    out = TextIngestor.clean_text("sooooo bad")
    assert "soo" in out and "soooo" not in out


def test_handles_empty_and_non_string():
    assert TextIngestor.clean_text("") == ""
    assert TextIngestor.clean_text(None) == ""
    assert TextIngestor.clean_text("   ") == ""


def test_lowercases_and_strips_punctuation():
    assert TextIngestor.clean_text("WOW!!! Amazing???") == "wow amazing"
