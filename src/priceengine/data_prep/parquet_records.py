"""On-disk records for data prep: ``ProductListing`` and ``EvalItem`` parquet.

This module is the persistence boundary for typed rows. It does not talk to
Hugging Face — ``dataset_builder.py`` does that — and it does not know about prompts or
metrics. Training and eval both import loaders from here (via ``data_prep``).

Layout produced / consumed elsewhere in the repo:

* ``data/splits/{train,val,test}.parquet`` — ``ProductListing`` rows for SFT
* ``data/golden/amazon.parquet`` — ``EvalItem`` rows for the fair-eval leaderboard
* ``data/combined/amazon.parquet`` — optional combined ``ProductListing`` dump
* ``reports/*.json`` — small prep metadata written next to leaderboards

``ProductListing`` vs ``EvalItem``: listings are the training/split schema
(``list_price``, ``item_id``). Eval items are the scoring schema (``price``,
``id``) used by pricers. ``products_to_eval_items`` is the projection between them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from priceengine.models import EvalItem, ProductListing


def save_products(path: Path, products: list[ProductListing]) -> None:
    """Write ``ProductListing`` rows to a parquet file (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([p.model_dump(mode="json") for p in products]).to_parquet(
        path, index=False
    )


def load_products(path: Path) -> list[ProductListing]:
    """Load ``ProductListing`` rows from parquet (e.g. a train/val/test split)."""
    frame = pd.read_parquet(path)
    return [
        ProductListing.model_validate(row) for row in frame.to_dict(orient="records")
    ]


def save_eval_items(path: Path, items: list[EvalItem]) -> None:
    """Write ``EvalItem`` rows to parquet (typically the golden set)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item.model_dump(mode="json") for item in items]).to_parquet(
        path, index=False
    )


def load_eval_items(path: Path) -> list[EvalItem]:
    """Load ``EvalItem`` rows from parquet for the eval leaderboard."""
    frame = pd.read_parquet(path)
    return [EvalItem.model_validate(row) for row in frame.to_dict(orient="records")]


def save_json(path: Path, payload: dict) -> None:
    """Write a small JSON report (prep counts, token-budget tables, etc.)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def products_to_eval_items(
    products: list[ProductListing],
    *,
    source: str = "list_price",
) -> list[EvalItem]:
    """Project training/split rows into the eval scoring schema.

    Field rename map: ``item_id``→``id``, ``list_price``→``price``. ``source``
    records that labels are Amazon list prices (not sold comps).
    """
    return [
        EvalItem(
            id=product.item_id,
            title=product.title,
            description=product.description,
            condition=product.condition,
            category=product.category,
            price=product.list_price,
            source=source,
        )
        for product in products
    ]
