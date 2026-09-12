"""Regex-based text sanitization for raw multi-channel customer feedback."""
import re


class TextIngestor:
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _URL_RE = re.compile(r"https?://\S+|www\.\S+")
    _HASHTAG_RE = re.compile(r"#(\w+)")
    _MENTION_RE = re.compile(r"@(\w+)")
    _NON_ALNUM_RE = re.compile(r"[^a-z0-9\s']")
    _WHITESPACE_RE = re.compile(r"\s+")
    _REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")

    @classmethod
    def extract_hashtags(cls, text: str) -> list:
        if not isinstance(text, str):
            return []
        return cls._HASHTAG_RE.findall(text)

    @classmethod
    def extract_mentions(cls, text: str) -> list:
        if not isinstance(text, str):
            return []
        return cls._MENTION_RE.findall(text)

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Strip HTML/URLs/mentions/special chars, collapse repeats & whitespace, lowercase."""
        if not isinstance(text, str) or not text.strip():
            return ""
        t = text.lower()
        t = cls._HTML_TAG_RE.sub(" ", t)
        t = cls._URL_RE.sub(" ", t)
        t = cls._MENTION_RE.sub(" ", t)
        t = cls._HASHTAG_RE.sub(r"\1", t)  # keep hashtag word, drop the '#'
        t = cls._REPEAT_CHAR_RE.sub(r"\1\1", t)  # "sooooo" -> "soo"
        t = cls._NON_ALNUM_RE.sub(" ", t)
        t = cls._WHITESPACE_RE.sub(" ", t).strip()
        return t
