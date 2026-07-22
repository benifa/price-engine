"""Parquet / JSONL IO for cleaned corpora and splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from priceengine.models import EvalItem, SoldListing


def listings_to_frame(listings: list[SoldListing]) -> pd.DataFrame:
    return pd.DataFrame([listing.model_dump(mode="json") for listing in listings])


def frame_to_listings(frame: pd.DataFrame) -> list[SoldListing]:
    return [SoldListing.model_validate(row) for row in frame.to_dict(orient="records")]


def save_listings(path: Path, listings: list[SoldListing]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    listings_to_frame(listings).to_parquet(path, index=False)


def load_listings(path: Path) -> list[SoldListing]:
    return frame_to_listings(pd.read_parquet(path))


def save_eval_items(path: Path, items: list[EvalItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item.model_dump(mode="json") for item in items]).to_parquet(path, index=False)


def load_eval_items(path: Path) -> list[EvalItem]:
    frame = pd.read_parquet(path)
    return [EvalItem.model_validate(row) for row in frame.to_dict(orient="records")]


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
