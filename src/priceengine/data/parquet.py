"""Read/write ProductListing and EvalItem parquet files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from priceengine.models import EvalItem, ProductListing


def save_products(path: Path, products: list[ProductListing]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([p.model_dump(mode="json") for p in products]).to_parquet(
        path, index=False
    )


def load_products(path: Path) -> list[ProductListing]:
    frame = pd.read_parquet(path)
    return [
        ProductListing.model_validate(row) for row in frame.to_dict(orient="records")
    ]


def save_eval_items(path: Path, items: list[EvalItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item.model_dump(mode="json") for item in items]).to_parquet(
        path, index=False
    )


def load_eval_items(path: Path) -> list[EvalItem]:
    frame = pd.read_parquet(path)
    return [EvalItem.model_validate(row) for row in frame.to_dict(orient="records")]


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def products_to_eval_items(
    products: list[ProductListing],
    *,
    source: str = "list_price",
) -> list[EvalItem]:
    """Convert split rows into the eval scoring schema."""
    return [
        EvalItem(
            item_id=product.item_id,
            title=product.title,
            description=product.description,
            condition=product.condition,
            category=product.category,
            price=product.price,
            source=source,
        )
        for product in products
    ]
