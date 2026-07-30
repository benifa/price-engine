"""Tests for eval role / short-label helpers."""

from priceengine.eval.baseline_pricer import BASELINE_NAME
from priceengine.eval.pricers import (
    OVERALL_TRAIN_MEDIAN_NAME,
    SAME_CATEGORY_MEDIAN_NAME,
)
from priceengine.eval.roles import (
    ROLE_FRONTIER,
    ROLE_NAIVE,
    ROLE_OURS,
    ROLE_SPECIALIST,
    role_for,
    short_label,
)


def test_roles_for_known_names():
    assert role_for(SAME_CATEGORY_MEDIAN_NAME) == ROLE_NAIVE
    assert role_for(OVERALL_TRAIN_MEDIAN_NAME) == ROLE_NAIVE
    assert role_for(BASELINE_NAME) == ROLE_SPECIALIST
    assert role_for("Frontier · gpt-5") == ROLE_FRONTIER
    assert role_for("list_price_qlora") == ROLE_OURS


def test_short_labels():
    assert short_label(SAME_CATEGORY_MEDIAN_NAME) == "Category median"
    assert short_label(OVERALL_TRAIN_MEDIAN_NAME) == "Train median"
    assert "specialist" in short_label(BASELINE_NAME).lower()
