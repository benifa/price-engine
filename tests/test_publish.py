"""Unit tests for Hub publish helpers (no network)."""

import pytest

from priceengine.train.publish import build_model_card, normalize_tag


def test_normalize_tag_semver():
    assert normalize_tag("0.1.0") == "v0.1.0"
    assert normalize_tag("v0.1.0") == "v0.1.0"


def test_normalize_tag_date():
    assert normalize_tag("2026-07-28") == "2026-07-28"


def test_normalize_tag_rejects_junk():
    with pytest.raises(ValueError):
        normalize_tag("latest")


def test_build_model_card_includes_load_snippet(tmp_path):
    card = build_model_card(
        repo_id="benifa/list-price-qlora",
        tag="v0.1.0",
        base_model="meta-llama/Llama-3.2-3B",
        adapter_path=tmp_path,
        leaderboard_md=None,
        private=True,
    )
    assert "PeftModel.from_pretrained" in card
    assert "revision=\"v0.1.0\"" in card
    assert "Price is $" in card
