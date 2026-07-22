"""Time-based train/val/test splits with near-duplicate leakage controls."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date, timedelta

from priceengine.models import EvalItem, SoldListing


def _normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def near_dupe_key(listing: SoldListing) -> str:
    return f"{_normalize_title(listing.title)}|{listing.condition}"


def time_split(
    listings: list[SoldListing],
    *,
    as_of: date | None = None,
    val_days: int = 7,
    test_days: int = 7,
) -> tuple[list[SoldListing], list[SoldListing], list[SoldListing]]:
    """Split by sold_date: train < T-14d, val [T-14d, T-7d), test [T-7d, T]."""
    if not listings:
        return [], [], []
    as_of = as_of or max(listing.sold_date for listing in listings)
    test_start = as_of - timedelta(days=test_days - 1)
    val_start = test_start - timedelta(days=val_days)

    train, val, test = [], [], []
    for listing in listings:
        if listing.sold_date >= test_start:
            test.append(listing)
        elif listing.sold_date >= val_start:
            val.append(listing)
        else:
            train.append(listing)
    return train, val, test


def remove_boundary_near_dupes(
    train: list[SoldListing],
    val: list[SoldListing],
    test: list[SoldListing],
) -> tuple[list[SoldListing], list[SoldListing], list[SoldListing], int]:
    """Drop val/test items whose near-dupe key appears in an earlier split."""
    train_keys = {near_dupe_key(x) for x in train}
    val_kept = [x for x in val if near_dupe_key(x) not in train_keys]
    dropped = len(val) - len(val_kept)
    earlier = train_keys | {near_dupe_key(x) for x in val_kept}
    test_kept = [x for x in test if near_dupe_key(x) not in earlier]
    dropped += len(test) - len(test_kept)
    return train, val_kept, test_kept, dropped


def stratify_report(listings: list[SoldListing]) -> dict:
    """Category × condition counts for docs/reports."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for listing in listings:
        counts[listing.category][listing.condition] += 1
    return {cat: dict(cond) for cat, cond in counts.items()}


def to_eval_items(listings: list[SoldListing], *, source: str = "sold") -> list[EvalItem]:
    return [
        EvalItem(
            id=listing.item_id,
            title=listing.title,
            description=listing.description,
            condition=listing.condition,
            category=listing.category,
            sold_date=listing.sold_date,
            price=listing.sold_price,
            source=source,
        )
        for listing in listings
    ]
