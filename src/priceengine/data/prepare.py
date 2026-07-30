"""Hub → local parquet splits + golden set.

Step 1 of the loop::

    Hub datasets  →  data/splits/  +  data/golden/amazon.parquet

* lite — small official splits; golden = lite test
* full — large subsample; golden is still lite test (those titles held out of train/val)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from priceengine.config import PRICE_MAX, PRICE_MIN, Settings
from priceengine.data.parquet import (
    products_to_eval_items,
    save_eval_items,
    save_json,
    save_products,
)
from priceengine.models import ProductListing

logger = logging.getLogger(__name__)

LITE_DATASET_ID = "benifa/items_lite"
FULL_DATASET_ID = "benifa/items_full"
_DATASET_BY_SIZE = {"lite": LITE_DATASET_ID, "full": FULL_DATASET_ID}
_HUB_SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))


def clamp_usd_price(price: float) -> float:
    """Round to nearest dollar and clamp to [$1, $999] (single Llama token)."""
    return float(max(PRICE_MIN, min(PRICE_MAX, round(price))))


def hub_row_to_product(
    row: dict[str, Any],
    *,
    split: str,
    index: int,
    id_prefix: str = "amazon",
) -> ProductListing | None:
    """Map one Hub row → ProductListing, or None if unusable."""
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

    item_id = str(row.get("id") or f"{id_prefix}:{split}:{index}")
    return ProductListing(
        item_id=item_id,
        title=title or summary[:120],
        description=summary,
        category=str(row.get("category") or "Other"),
        price=clamp_usd_price(price),
        condition="new",
    )


def prepare_dataset(
    settings: Settings,
    *,
    size: str = "lite",
    dataset_id: str | None = None,
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> dict[str, int]:
    """Download from Hub and write splits + golden. Returns row counts."""
    size = size.lower().strip()
    if size not in _DATASET_BY_SIZE:
        raise ValueError(f"size must be 'lite' or 'full', got {size!r}")

    hub_id = dataset_id or _DATASET_BY_SIZE[size]
    logger.info("Downloading %s (size=%s)", hub_id, size)

    if size == "lite":
        return _build_lite(settings, hub_id=hub_id, train_limit=train_limit)
    return _build_full(
        settings,
        hub_id=hub_id,
        train_limit=train_limit or 100_000,
        val_limit=val_limit or 2_000,
    )


def _map_rows(
    rows: Iterable[Any],
    *,
    split: str,
    id_prefix: str,
    limit: int | None = None,
    skip_titles: set[str] | None = None,
) -> list[ProductListing]:
    blocked = skip_titles or set()
    out: list[ProductListing] = []
    for i, row in enumerate(rows):
        if limit is not None and len(out) >= limit:
            break
        as_dict = dict(row)
        title = (as_dict.get("title") or "").strip()
        if title in blocked:
            continue
        product = hub_row_to_product(
            as_dict, split=split, index=i, id_prefix=id_prefix
        )
        if product is not None:
            out.append(product)
    return out


def _write_split(settings: Settings, name: str, products: list[ProductListing]) -> None:
    save_products(settings.splits_dir / f"{name}.parquet", products)


def _build_lite(
    settings: Settings,
    *,
    hub_id: str,
    train_limit: int | None,
) -> dict[str, int]:
    from datasets import load_dataset

    online = load_dataset(hub_id)
    counts: dict[str, int] = {}
    golden: list[ProductListing] = []

    for hub_name, local_name in _HUB_SPLITS:
        rows = online[hub_name]
        if local_name == "train" and train_limit:
            rows = rows.select(range(min(train_limit, len(rows))))

        products = _map_rows(rows, split=local_name, id_prefix="amazon_lite")
        _write_split(settings, local_name, products)
        counts[local_name] = len(products)
        if local_name == "test":
            golden = products
        logger.info("Wrote %s: %d products", local_name, len(products))

    save_eval_items(
        settings.golden_dir / "amazon.parquet",
        products_to_eval_items(golden),
    )
    save_json(
        settings.reports_dir / "amazon_prep.json",
        {
            "size": "lite",
            "dataset": hub_id,
            "counts": counts,
            "note": "Golden = lite test. Labels = Amazon list prices.",
        },
    )
    return counts


def _build_full(
    settings: Settings,
    *,
    hub_id: str,
    train_limit: int,
    val_limit: int,
) -> dict[str, int]:
    """Large train/val subsample; golden = lite test (titles excluded from train/val)."""
    from datasets import load_dataset

    large = load_dataset(hub_id)
    lite = load_dataset(LITE_DATASET_ID)
    holdout_titles = {title for title in lite["test"]["title"] if title}
    logger.info("Excluding %d golden titles from full train/val", len(holdout_titles))

    train = _map_rows(
        large["train"],
        split="train",
        id_prefix="amazon_full",
        limit=train_limit,
        skip_titles=holdout_titles,
    )
    val = _map_rows(
        large["validation"],
        split="val",
        id_prefix="amazon_full",
        limit=val_limit,
        skip_titles=holdout_titles,
    )
    test_peek = _map_rows(
        large["test"],
        split="test",
        id_prefix="amazon_full",
        limit=min(2_000, len(large["test"])),
    )

    _write_split(settings, "train", train)
    _write_split(settings, "val", val)
    _write_split(settings, "test", test_peek)

    golden = _map_rows(lite["test"], split="test", id_prefix="amazon_lite")
    save_eval_items(
        settings.golden_dir / "amazon.parquet",
        products_to_eval_items(golden),
    )

    counts = {
        "train": len(train),
        "val": len(val),
        "test_peek": len(test_peek),
        "golden": len(golden),
    }
    save_json(
        settings.reports_dir / "amazon_prep.json",
        {
            "size": "full",
            "dataset": hub_id,
            "train_limit": train_limit,
            "val_limit": val_limit,
            "holdout_titles": len(holdout_titles),
            "counts": counts,
            "note": "Train/val from full subsample; golden = lite test.",
        },
    )
    logger.info("Full build done: %s", counts)
    return counts
