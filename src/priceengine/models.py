"""Shared data models for data prep, training, and evaluation.

* ``ProductListing`` — one training/split row (``list_price``, ``item_id``)
* ``EvalItem`` — one golden-set row for scoring (``price``, ``id``)
* ``Prediction`` / ``RunMetrics`` / ``ComparisonResult`` — eval outputs
"""

from pydantic import BaseModel, Field

from priceengine.config import Condition


class ProductListing(BaseModel):
    """One product with a known list price (training / split row)."""

    item_id: str
    title: str
    description: str
    category: str
    list_price: float
    condition: Condition = "new"

    def text_for_pricing(self) -> str:
        desc = self.description.strip()
        if desc and desc.lower() != self.title.lower():
            return f"{self.title.strip()}\n{desc}"
        return self.title.strip()


class EvalItem(BaseModel):
    """One held-out example used when scoring a pricer."""

    id: str
    title: str
    description: str
    price: float = Field(description="Ground-truth list price in USD")
    category: str | None = None
    condition: Condition | None = "new"
    truncated: bool = False
    source: str = "list_price"

    def text_for_pricing(self) -> str:
        desc = self.description.strip()
        if desc and desc.lower() != self.title.lower():
            return f"{self.title.strip()}\n{desc}"
        return self.title.strip()


class Prediction(BaseModel):
    """One pricer output for an EvalItem."""

    item_id: str
    guess: float
    truth: float
    error: float
    truncated: bool = False
    category: str | None = None
    condition: str | None = None


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
    """Paired comparison of a challenger vs a baseline pricer."""

    challenger: str
    baseline: str
    eval_set: str
    delta_mae: float  # baseline_mae - challenger_mae (positive => challenger wins)
    relative_improvement: float
    ci_low: float
    ci_high: float
    n: int
    victory: bool


__all__ = [
    "ProductListing",
    "EvalItem",
    "Prediction",
    "RunMetrics",
    "ComparisonResult",
    "Condition",
]
