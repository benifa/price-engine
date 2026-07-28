"""Central configuration for price-engine."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# List-price labels are treated as new-condition catalog prices (not sold comps).
Condition = Literal["new"]

# Llama tokenizers encode integers 0–999 as a single token — keep labels in range.
PRICE_MIN = 1.0
PRICE_MAX = 999.0

AMAZON_LIST_QUESTION = "What does this cost to the nearest dollar?"
PRICE_PREFIX = "Price is $"
SUMMARY_CUTOFF = 110

BASE_MODEL = "meta-llama/Llama-3.2-3B"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRICEENGINE_", env_file=".env", extra="ignore"
    )

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")

    # Eval defaults (see docs/COMPARISON.md)
    bootstrap_samples: int = 10_000
    hit_abs_dollars: float = 40.0
    hit_rel_fraction: float = 0.20
    victory_relative_mae: float = 0.25  # 25% relative MAE improvement vs published baseline

    @property
    def combined_dir(self) -> Path:
        """Optional union dump of all ProductListing rows (``data/combined/``)."""
        return self.data_dir / "combined"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def golden_dir(self) -> Path:
        return self.data_dir / "golden"


def get_settings() -> Settings:
    return Settings()
