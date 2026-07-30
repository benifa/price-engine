"""Human labels for leaderboard / HTML report rows.

Internal prediction keys stay stable for caching; this module only affects how we
explain each row (our model vs frontier vs specialist vs naive floor).
"""

from __future__ import annotations

from priceengine.eval.baseline_pricer import BASELINE_NAME
from priceengine.eval.pricers import (
    OVERALL_TRAIN_MEDIAN_NAME,
    SAME_CATEGORY_MEDIAN_NAME,
)

ROLE_OURS = "Our model"
ROLE_FRONTIER = "Frontier API"
ROLE_SPECIALIST = "Published specialist"
ROLE_NAIVE = "Naive floor"
ROLE_OTHER = "Other"

_MEDIAN_NAMES = frozenset({SAME_CATEGORY_MEDIAN_NAME, OVERALL_TRAIN_MEDIAN_NAME})
# Legacy keys from older leaderboard.json / screenshots
_LEGACY_MEDIANS = frozenset(
    {"Same-category train median", "Overall train median"}
)

_SHORT: dict[str, str] = {
    SAME_CATEGORY_MEDIAN_NAME: "Category median",
    OVERALL_TRAIN_MEDIAN_NAME: "Train median",
    "Same-category train median": "Category median",
    "Overall train median": "Train median",
    BASELINE_NAME: "ed-donner specialist",
}

_BLURB: dict[str, str] = {
    SAME_CATEGORY_MEDIAN_NAME: (
        "Guess the median train price in the same product category "
        "(no model — sanity floor)."
    ),
    OVERALL_TRAIN_MEDIAN_NAME: (
        "Always guess one number: the median of all train prices "
        "(no model — weakest floor)."
    ),
    "Same-category train median": (
        "Guess the median train price in the same product category "
        "(no model — sanity floor)."
    ),
    "Overall train median": (
        "Always guess one number: the median of all train prices "
        "(no model — weakest floor)."
    ),
    BASELINE_NAME: (
        "Published Amazon list-price specialist "
        "(ed-donner/price-2025-11-28 via Modal)."
    ),
}


def is_naive_floor(name: str) -> bool:
    return name in _MEDIAN_NAMES or name in _LEGACY_MEDIANS


def is_frontier(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("frontier") or "gpt-" in lowered or "claude" in lowered


def is_specialist(name: str) -> bool:
    return name == BASELINE_NAME or "ed-donner" in name.lower()


def role_for(name: str) -> str:
    if is_naive_floor(name):
        return ROLE_NAIVE
    if is_specialist(name):
        return ROLE_SPECIALIST
    if is_frontier(name):
        return ROLE_FRONTIER
    # Fine-tuned challengers use adapter path / --name.
    return ROLE_OURS


def short_label(name: str) -> str:
    if name in _SHORT:
        return _SHORT[name]
    if name.lower().startswith("frontier · "):
        return name.split("·", 1)[1].strip()
    return name


def blurb_for(name: str) -> str:
    if name in _BLURB:
        return _BLURB[name]
    role = role_for(name)
    if role == ROLE_FRONTIER:
        return "General-purpose frontier chat model prompted for a dollar price."
    if role == ROLE_OURS:
        return "Our fine-tuned list-price adapter (challenger)."
    return ""


def role_color(role: str) -> str:
    return {
        ROLE_OURS: "#2563eb",
        ROLE_FRONTIER: "#7c3aed",
        ROLE_SPECIALIST: "#0d9488",
        ROLE_NAIVE: "#94a3b8",
        ROLE_OTHER: "#64748b",
    }.get(role, "#64748b")
