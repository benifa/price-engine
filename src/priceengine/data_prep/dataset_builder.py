"""Build local train/val/test + golden datasets from Hugging Face Hub.

This is step 1 of the project (see ``docs/DESIGN.md``):

    Hub Amazon list-price datasets  →  ``data/splits/``, ``data/combined/``, ``data/golden/``

Downstream consumers:

* ``training/`` — reads ``data/splits/*.parquet`` to build prompt/completion examples
* ``eval/`` — scores pricers on ``data/golden/amazon.parquet`` (the held-out golden set)

Two sizes are supported:

* **lite** — small official Hub splits; golden set = the lite *test* split
* **full** — subsample of the large Hub train/val; golden set is *still* lite test,
  and those titles are excluded from train/val so eval is not contaminated
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from priceengine.config import PRICE_MAX, PRICE_MIN, Settings
from priceengine.data_prep.parquet_records import (
    products_to_eval_items,
    save_eval_items,
    save_json,
    save_products,
)
from priceengine.models import ProductListing

logger = logging.getLogger(__name__)

# Published mirrors under the project owner's Hub account (see README).
HF_USER = "benifa"
LITE_DATASET_ID = f"{HF_USER}/items_lite"
FULL_DATASET_ID = f"{HF_USER}/items_full"
_DATASET_BY_SIZE = {"lite": LITE_DATASET_ID, "full": FULL_DATASET_ID}

# Hub split name → local filename stem under ``data/splits/``.
_HUB_SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))


def clamp_usd_price(price: float) -> float:
    """Round to the nearest dollar and clamp to [$PRICE_MIN, $PRICE_MAX].

    Llama tokenizers encode integers 0–999 as a *single* token. Keeping labels in
    that range makes the completion ``Price is $NNN.00`` cheap and stable to learn.
    """
    return float(max(PRICE_MIN, min(PRICE_MAX, round(price))))


def hub_row_to_product(
    row: dict[str, Any],
    *,
    split: str,
    index: int,
    id_prefix: str = "amazon",
) -> ProductListing | None:
    """Map one Hub row to a ``ProductListing``, or ``None`` if it is unusable.

    Expected Hub fields: ``title``, ``summary`` (description text), ``price``,
    optional ``category`` / ``id``. Rows outside [$PRICE_MIN, $PRICE_MAX] or with
    empty text are dropped rather than written into training data.
    """
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

    # Prefer the publisher id when present so golden-set ids stay stable across runs.
    item_id = str(row.get("id") or f"{id_prefix}:{split}:{index}")
    return ProductListing(
        item_id=item_id,
        title=title or summary[:120],
        description=summary,
        category=str(row.get("category") or "Other"),
        list_price=clamp_usd_price(price),
        # Labels are Amazon *list* prices, not marketplace sold comps — treat as new.
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
    """Download from Hub and write the local parquet layout.

    Returns a dict of row counts (useful for CLI / prep reports).

    Parameters
    ----------
    size:
        ``lite`` keeps the publisher's train/val/test splits.
        ``full`` subsamples the large dataset but always uses lite test as golden.
    dataset_id:
        Optional Hub override; defaults to ``benifa/items_lite`` or ``items_full``.
    train_limit / val_limit:
        Caps for the full path (and optional train cap on lite).
    """
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
    """Convert many Hub rows, optionally stopping early or skipping titles."""
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
    """Write one split parquet under ``data/splits/{name}.parquet``."""
    save_products(settings.splits_dir / f"{name}.parquet", products)


def _build_lite(
    settings: Settings,
    *,
    hub_id: str,
    train_limit: int | None,
) -> dict[str, int]:
    """Materialize the small dataset: local splits mirror Hub; golden = test."""
    from datasets import load_dataset

    online = load_dataset(hub_id)
    counts: dict[str, int] = {}
    all_products: list[ProductListing] = []
    golden: list[ProductListing] = []

    for hub_name, local_name in _HUB_SPLITS:
        rows = online[hub_name]
        if local_name == "train" and train_limit:
            rows = rows.select(range(min(train_limit, len(rows))))

        products = _map_rows(rows, split=local_name, id_prefix="amazon_lite")
        _write_split(settings, local_name, products)
        counts[local_name] = len(products)
        all_products.extend(products)
        if local_name == "test":
            golden = products
        logger.info("Wrote %s: %d products", local_name, len(products))

    # Combined dump is handy for ad-hoc inspection; training uses splits/.
    save_products(settings.combined_dir / "amazon.parquet", all_products)
    # Golden is EvalItem-shaped so the eval leaderboard can load it directly.
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
    """Materialize a large train/val subsample with lite-test as the golden holdout.

    Fair-eval rule: any title that appears in the golden set must not appear in
    train or val, otherwise MAE on golden can leak from memorization.
    """
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
    # A small peek of full-test is written for sanity checks; it is *not* the golden set.
    test_peek = _map_rows(
        large["test"],
        split="test",
        id_prefix="amazon_full",
        limit=min(2_000, len(large["test"])),
    )

    _write_split(settings, "train", train)
    _write_split(settings, "val", val)
    _write_split(settings, "test", test_peek)
    save_products(settings.combined_dir / "amazon.parquet", train + val + test_peek)

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
            "note": (
                "Train/val from full subsample; golden = lite test. "
                "Golden titles excluded from train/val."
            ),
        },
    )
    logger.info("Full build done: %s", counts)
    return counts
