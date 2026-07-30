"""Shared data models.

* ProductListing — one row in data/splits/
* EvalItem — one row in data/golden/ (what we score)
* Prediction — one model estimate vs actual price
* RunMetrics / ComparisonResult — leaderboard aggregates
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from priceengine.config import Condition


def product_text(title: str, description: str) -> str:
    """Title + description for the pricing prompt (skip duplicate description)."""
    title = title.strip()
    description = description.strip()
    if description and description.lower() != title.lower():
        return f"{title}\n{description}"
    return title


class ProductListing(BaseModel):
    """One product with a known list price (train/val/test split row)."""

    item_id: str
    title: str
    description: str
    category: str
    price: float
    condition: Condition = "new"

    @model_validator(mode="before")
    @classmethod
    def _accept_old_list_price_column(cls, data: Any) -> Any:
        # Older parquet used list_price instead of price.
        if isinstance(data, dict) and "price" not in data and "list_price" in data:
            data = {**data, "price": data["list_price"]}
        return data

    def text_for_pricing(self) -> str:
        return product_text(self.title, self.description)


class EvalItem(BaseModel):
    """One held-out product used when scoring a pricer."""

    item_id: str
    title: str
    description: str
    price: float = Field(description="Ground-truth list price in USD")
    category: str | None = None
    condition: Condition | None = "new"
    truncated: bool = False
    source: str = "list_price"

    @model_validator(mode="before")
    @classmethod
    def _accept_old_id_column(cls, data: Any) -> Any:
        # Older golden parquet used id instead of item_id.
        if isinstance(data, dict) and "item_id" not in data and "id" in data:
            data = {**data, "item_id": data["id"]}
        return data

    def text_for_pricing(self) -> str:
        return product_text(self.title, self.description)


class Prediction(BaseModel):
    """One pricer output for one EvalItem."""

    item_id: str
    estimate: float  # model guess
    actual: float  # ground-truth price
    error: float  # abs(estimate - actual)
    truncated: bool = False
    category: str | None = None
    condition: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_old_guess_truth(cls, data: Any) -> Any:
        # Older leaderboard JSON used guess/truth.
        if isinstance(data, dict):
            if "estimate" not in data and "guess" in data:
                data = {**data, "estimate": data["guess"]}
            if "actual" not in data and "truth" in data:
                data = {**data, "actual": data["truth"]}
        return data


class RunMetrics(BaseModel):
    """Aggregate metrics for one pricer on one eval set."""

    name: str
    eval_set: str
    n: int
    mae: float
    median_ape: float
    hit_rate: float
    rmsle: float
    mae_ci_low: float | None = None
    mae_ci_high: float | None = None


class ComparisonResult(BaseModel):
    """Paired comparison of challenger vs baseline (positive delta_mae = challenger better)."""

    challenger: str
    baseline: str
    eval_set: str
    delta_mae: float
    relative_improvement: float
    ci_low: float
    ci_high: float
    n: int
    victory: bool
