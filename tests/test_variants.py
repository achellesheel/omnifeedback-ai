import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.variants import DEFAULT_VARIANT, VARIANTS


def test_all_variants_registered():
    assert set(VARIANTS.keys()) == {"v1_naive", "v2_hardened", "v3_transformer"}


def test_default_variant_is_v3():
    assert DEFAULT_VARIANT == "v3_transformer"


def test_v1_v2_have_distinct_paths():
    v1, v2 = VARIANTS["v1_naive"], VARIANTS["v2_hardened"]
    assert v1["db_path"] != v2["db_path"]
    assert v1["model_dir"] != v2["model_dir"]
    assert v1["raw_csv"] != v2["raw_csv"]


def test_v3_shares_v2_data_but_has_own_model_dir():
    v2, v3 = VARIANTS["v2_hardened"], VARIANTS["v3_transformer"]
    assert v2["raw_csv"] == v3["raw_csv"]
    assert v2["db_path"] == v3["db_path"]
    assert v2["model_dir"] != v3["model_dir"]


def test_required_keys_present():
    for vid, v in VARIANTS.items():
        for key in ("label", "short_label", "description", "raw_csv", "db_path", "model_dir", "model_type"):
            assert key in v, f"{vid} missing {key}"


def test_model_types_are_known():
    for vid, v in VARIANTS.items():
        assert v["model_type"] in ("bilstm", "transformer"), f"{vid} has unknown model_type {v['model_type']}"
