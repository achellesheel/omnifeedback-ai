import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.variants import DEFAULT_VARIANT, VARIANTS


def test_both_variants_registered():
    assert set(VARIANTS.keys()) == {"v1_naive", "v2_hardened"}


def test_default_variant_is_hardened():
    assert DEFAULT_VARIANT == "v2_hardened"


def test_variants_have_distinct_paths():
    v1, v2 = VARIANTS["v1_naive"], VARIANTS["v2_hardened"]
    assert v1["db_path"] != v2["db_path"]
    assert v1["model_dir"] != v2["model_dir"]
    assert v1["raw_csv"] != v2["raw_csv"]


def test_required_keys_present():
    for vid, v in VARIANTS.items():
        for key in ("label", "short_label", "description", "raw_csv", "db_path", "model_dir"):
            assert key in v, f"{vid} missing {key}"
