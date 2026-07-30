"""Shared knobs for data prep, training, and eval."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# All labels are Amazon-style list prices (catalog "new"), not sold comps.
Condition = Literal["new"]

# Llama encodes integers 0–999 as one token — keep training labels in this range.
PRICE_MIN = 1.0
PRICE_MAX = 999.0

# Prompt pieces (must match Hub rows and the published baseline).
PRICE_QUESTION = "What does this cost to the nearest dollar?"
PRICE_PREFIX = "Price is $"
MAX_DESCRIPTION_TOKENS = 110

BASE_MODEL = "meta-llama/Llama-3.2-3B"


class Settings(BaseSettings):
    """Paths and eval thresholds (overridable with PRICEENGINE_* env vars)."""

    model_config = SettingsConfigDict(
        env_prefix="PRICEENGINE_", env_file=".env", extra="ignore"
    )

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")

    # See docs/EVAL.md (fair comparison / victory thresholds)
    bootstrap_samples: int = 10_000
    hit_abs_dollars: float = 40.0
    hit_rel_fraction: float = 0.20
    victory_relative_mae: float = 0.25

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def golden_dir(self) -> Path:
        return self.data_dir / "golden"


def get_settings() -> Settings:
    return Settings()
