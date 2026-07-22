"""Cleaning rules for sold-listing corpora."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import date, datetime

from priceengine.config import PRICE_MAX, PRICE_MIN
from priceengine.corpus.conditions import normalize_condition
from priceengine.models import ListingFormat, SoldListing

_THIN_DESC = 20  # characters
_FLOOR_AUCTION = 5.0  # drop absurd auction endings below this


def _parse_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y"):
        try:
            return datetime.strptime(text[:10] if fmt == "%Y-%m-%d" else text, fmt).date()
        except ValueError:
            continue
    return None


def _listing_format(raw: str | None) -> ListingFormat:
    if not raw:
        return "unknown"
    lower = raw.lower()
    if "auction" in lower:
        return "auction"
    if "bin" in lower or "fixed" in lower or "buy" in lower:
        return "bin"
    return "unknown"


def listing_from_apify_item(item: dict) -> SoldListing | None:
    """Best-effort parse of a heterogeneous Apify eBay actor item into SoldListing."""
    title = (item.get("title") or item.get("name") or "").strip()
    if not title:
        return None
    description = (
        item.get("description")
        or item.get("shortDescription")
        or item.get("subtitle")
        or title
    )
    description = re.sub(r"<[^>]+>", " ", str(description))
    description = re.sub(r"\s+", " ", description).strip()

    price_raw = (
        item.get("soldPrice")
        or item.get("price")
        or item.get("currentPrice")
        or item.get("priceValue")
    )
    if isinstance(price_raw, dict):
        price_raw = price_raw.get("value") or price_raw.get("amount")
    try:
        price = float(str(price_raw).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None

    sold_date = _parse_date(
        item.get("soldDate")
        or item.get("endDate")
        or item.get("dateSold")
        or item.get("endedTime")
    )
    if sold_date is None:
        return None

    item_id = str(
        item.get("itemId") or item.get("id") or item.get("item_id") or f"{title}:{sold_date}"
    )
    condition = normalize_condition(
        item.get("conditionId") or item.get("condition") or item.get("conditionName")
    )
    category_raw = item.get("category") or item.get("categoryName") or item.get("categories")
    if isinstance(category_raw, list) and category_raw:
        category_raw = category_raw[0]
    if isinstance(category_raw, dict):
        category_raw = category_raw.get("name") or "Other"
    category = str(category_raw or "Other")

    url = str(item.get("url") or item.get("itemUrl") or item.get("link") or "")
    fmt = _listing_format(item.get("listingType") or item.get("buyingFormat"))

    return SoldListing(
        item_id=item_id,
        title=title[:300],
        description=description[:4000],
        condition=condition,
        category=category[:80],
        sold_price=price,
        sold_date=sold_date,
        listing_format=fmt,
        url=url,
    )


def is_clean(listing: SoldListing) -> tuple[bool, str]:
    """Return (ok, reason). reason is empty when ok."""
    if listing.sold_price < PRICE_MIN or listing.sold_price > PRICE_MAX:
        return False, "price_out_of_range"
    if listing.listing_format == "auction" and listing.sold_price < _FLOOR_AUCTION:
        return False, "auction_floor"
    if len(listing.text_for_pricing()) < _THIN_DESC:
        return False, "thin_description"
    if not math.isfinite(listing.sold_price):
        return False, "non_finite_price"
    return True, ""


def round_price(price: float) -> float:
    return float(max(PRICE_MIN, min(PRICE_MAX, round(price))))


def clean_listings(raw: list[SoldListing]) -> tuple[list[SoldListing], dict[str, int]]:
    """Apply cleaning rules; round prices; return cleaned list + drop counters."""
    drops: dict[str, int] = defaultdict(int)
    kept: list[SoldListing] = []
    seen_ids: set[str] = set()
    for listing in raw:
        ok, reason = is_clean(listing)
        if not ok:
            drops[reason] += 1
            continue
        if listing.item_id in seen_ids:
            drops["duplicate_id"] += 1
            continue
        seen_ids.add(listing.item_id)
        kept.append(
            listing.model_copy(update={"sold_price": round_price(listing.sold_price)})
        )
    return kept, dict(drops)


def drop_log_outliers(
    listings: list[SoldListing], *, z_thresh: float = 3.5
) -> tuple[list[SoldListing], int]:
    """Drop per-category log-price outliers (modified z-score on log1p price)."""
    by_cat: dict[str, list[SoldListing]] = defaultdict(list)
    for listing in listings:
        by_cat[listing.category].append(listing)

    kept: list[SoldListing] = []
    dropped = 0
    for group in by_cat.values():
        if len(group) < 20:
            kept.extend(group)
            continue
        logs = [math.log1p(x.sold_price) for x in group]
        median = sorted(logs)[len(logs) // 2]
        abs_dev = [abs(v - median) for v in logs]
        mad = sorted(abs_dev)[len(abs_dev) // 2] or 1e-6
        for listing, value in zip(group, logs):
            z = 0.6745 * (value - median) / mad
            if abs(z) > z_thresh:
                dropped += 1
            else:
                kept.append(listing)
    return kept, dropped
