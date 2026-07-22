"""Condition normalization from eBay (and similar) condition strings/IDs."""

from __future__ import annotations

import re

from priceengine.config import Condition

# eBay conditionId → our enum (common Browse/Finding API IDs)
EBAY_CONDITION_IDS: dict[int, Condition] = {
    1000: "new",
    1500: "new",  # new other
    1750: "new",  # new with defects (treat as new-ish; open-box if text says so)
    2000: "refurb",  # certified refurbished
    2010: "refurb",
    2020: "refurb",
    2030: "refurb",
    2500: "open-box",  # seller refurbished / open box
    2750: "used-good",  # like new
    3000: "used-good",
    4000: "used-fair",
    5000: "used-fair",
    6000: "for-parts",
    7000: "for-parts",
}

_PATTERNS: list[tuple[re.Pattern[str], Condition]] = [
    (re.compile(r"for\s*parts|not\s*working|as[\s-]*is", re.I), "for-parts"),
    (re.compile(r"refurb", re.I), "refurb"),
    (re.compile(r"open[\s-]*box", re.I), "open-box"),
    (re.compile(r"like\s*new|excellent|used[\s-]*good|pre[\s-]*owned", re.I), "used-good"),
    (re.compile(r"fair|acceptable|good\s*condition|used", re.I), "used-fair"),
    (re.compile(r"\bnew\b", re.I), "new"),
]


def normalize_condition(
    raw: str | int | None,
    *,
    fallback: Condition = "used-good",
) -> Condition:
    """Map an eBay condition id or free-text label to our enum."""
    if raw is None or raw == "":
        return fallback
    if isinstance(raw, int) or (isinstance(raw, str) and raw.isdigit()):
        return EBAY_CONDITION_IDS.get(int(raw), fallback)
    text = str(raw).strip()
    for pattern, label in _PATTERNS:
        if pattern.search(text):
            return label
    return fallback
