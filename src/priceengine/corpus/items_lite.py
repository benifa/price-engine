"""Load Ed Donner's items_lite dataset into our corpus / eval formats.

Use this path until Apify sold-listings data is available. Labels are Amazon
list prices of (mostly) new items — Ed's original target distribution.
Official HuggingFace train/validation/test splits are preserved (not time-split).
"""

from __future__ import annotations

import logging
from datetime import date

from priceengine.config import PRICE_MAX, PRICE_MIN, Settings
from priceengine.corpus.cleaning import round_price
from priceengine.corpus.io import save_eval_items, save_json, save_listings
from priceengine.corpus.splits import to_eval_items
from priceengine.models import SoldListing

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "ed-donner/items_lite"


def row_to_listing(row: dict, *, split: str, index: int) -> SoldListing | None:
    title = (row.get("title") or "").strip()
    summary = (row.get("summary") or title).strip()
    if not title and not summary:
        return None
    try:
        price = float(row["price"])
    except (KeyError, TypeError, ValueError):
        return None
    if price < PRICE_MIN or price > PRICE_MAX:
        return None
    item_id = str(row.get("id") or f"items_lite:{split}:{index}")
    # No real sold dates on items_lite; sentinel keeps the schema valid.
    return SoldListing(
        item_id=item_id,
        title=title or summary[:120],
        description=summary,
        condition="new",
        category=str(row.get("category") or "Other"),
        sold_price=round_price(price),
        sold_date=date(2024, 1, 1),
        listing_format="bin",
        url="",
    )


def prepare_items_lite(
    settings: Settings,
    *,
    dataset_name: str = DEFAULT_DATASET,
    train_limit: int | None = None,
) -> dict[str, int]:
    """Download items_lite and write clean/splits/golden under data/."""
    from datasets import load_dataset

    logger.info("Loading %s", dataset_name)
    ds = load_dataset(dataset_name)

    counts: dict[str, int] = {}
    all_clean: list[SoldListing] = []

    for split_name, hf_key in (("train", "train"), ("val", "validation"), ("test", "test")):
        rows = ds[hf_key]
        if split_name == "train" and train_limit:
            rows = rows.select(range(min(train_limit, len(rows))))
        listings: list[SoldListing] = []
        for i, row in enumerate(rows):
            listing = row_to_listing(dict(row), split=split_name, index=i)
            if listing is not None:
                listings.append(listing)
        save_listings(settings.splits_dir / f"{split_name}.parquet", listings)
        counts[split_name] = len(listings)
        all_clean.extend(listings)
        logger.info("%s: %d listings", split_name, len(listings))

    save_listings(settings.clean_dir / "sold.parquet", all_clean)

    from priceengine.corpus.io import load_listings

    test_listings = load_listings(settings.splits_dir / "test.parquet")
    golden = to_eval_items(test_listings, source="items_lite")
    for item in golden:
        item.sold_date = None
        item.condition = "new"
    save_eval_items(settings.golden_dir / "items_lite.parquet", golden)
    # Default CLI golden path during the Ed phase
    save_eval_items(settings.golden_dir / "used_goods.parquet", golden)

    save_json(
        settings.reports_dir / "items_lite_prep.json",
        {
            "dataset": dataset_name,
            "counts": counts,
            "note": (
                "Using Ed's official HF splits (not time-based). "
                "condition=new; labels=Amazon list prices."
            ),
        },
    )
    return counts
