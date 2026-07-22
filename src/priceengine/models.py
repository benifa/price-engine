"""Shared data models for corpus, training, and evaluation."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from priceengine.config import CONDITIONS, Condition

ListingFormat = Literal["auction", "bin", "unknown"]


class SoldListing(BaseModel):
    """One cleaned transactional record (eBay sold / completed)."""

    item_id: str
    title: str
    description: str
    condition: Condition
    category: str
    sold_price: float
    sold_date: date
    listing_format: ListingFormat = "unknown"
    url: str = ""

    def text_for_pricing(self) -> str:
        """Description blob passed to the model (title + description)."""
        desc = self.description.strip()
        if desc and desc.lower() != self.title.lower():
            return f"{self.title.strip()}\n{desc}"
        return self.title.strip()


class EvalItem(BaseModel):
    """A single evaluation example — identity-free description + ground truth."""

    id: str
    title: str
    description: str
    condition: Condition | None = None
    category: str | None = None
    sold_date: date | None = None
    price: float = Field(description="Ground-truth price (sold or list, depending on battleground)")
    truncated: bool = False
    source: str = "sold"  # "sold" | "items_lite"

    def text_for_pricing(self) -> str:
        desc = self.description.strip()
        if desc and desc.lower() != self.title.lower():
            return f"{self.title.strip()}\n{desc}"
        return self.title.strip()


class Prediction(BaseModel):
    """One model prediction for an EvalItem."""

    item_id: str
    guess: float
    truth: float
    error: float
    truncated: bool = False
    category: str | None = None
    condition: str | None = None


class RunMetrics(BaseModel):
    """Aggregate metrics for one contestant on one battleground."""

    name: str
    battleground: str
    n: int
    mae: float
    median_ape: float
    hit_rate: float
    rmsle: float
    mae_ci_low: float | None = None
    mae_ci_high: float | None = None


class ComparisonResult(BaseModel):
    """Paired comparison of our model vs a baseline (typically Ed)."""

    ours: str
    baseline: str
    battleground: str
    delta_mae: float  # baseline_mae - ours_mae (positive => we win)
    relative_improvement: float
    ci_low: float
    ci_high: float
    n: int
    victory: bool


# Keep CONDITIONS re-exported for callers that import from models
__all__ = [
    "SoldListing",
    "EvalItem",
    "Prediction",
    "RunMetrics",
    "ComparisonResult",
    "CONDITIONS",
    "Condition",
    "ListingFormat",
    "datetime",
]
