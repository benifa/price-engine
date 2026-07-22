"""Central configuration for price-engine."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Condition = Literal[
    "new",
    "open-box",
    "refurb",
    "used-good",
    "used-fair",
    "for-parts",
]

CONDITIONS: tuple[str, ...] = (
    "new",
    "open-box",
    "refurb",
    "used-good",
    "used-fair",
    "for-parts",
)

# Llama tokenizer encodes integers 0–999 as a single token — keep labels in range.
PRICE_MIN = 1.0
PRICE_MAX = 999.0

ED_QUESTION = "What does this cost to the nearest dollar?"
OURS_QUESTION = "How much did this sell for, to the nearest dollar?"
PRICE_PREFIX = "Price is $"

ED_MODEL_ID = "ed-donner/price-2025-11-28_18.47.07"
ED_BASE_MODEL = "meta-llama/Llama-3.2-3B"
ED_REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"

DEFAULT_CATEGORIES = [
    "Electronics",
    "Computers",
    "Tools",
    "Appliances",
    "Gaming",
    "Audio",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRICEENGINE_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")
    wandb_api_key: str = Field(default="", validation_alias="WANDB_API_KEY")
    wandb_project: str = Field(default="price-engine", validation_alias="WANDB_PROJECT")
    apify_token: str = Field(default="", validation_alias="APIFY_API_TOKEN")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    ebay_client_id: str = Field(default="", validation_alias="EBAY_CLIENT_ID")
    ebay_client_secret: str = Field(default="", validation_alias="EBAY_CLIENT_SECRET")

    # Eval defaults
    bootstrap_samples: int = 10_000
    hit_abs_dollars: float = 40.0
    hit_rel_fraction: float = 0.20
    victory_relative_mae: float = 0.25  # 25% relative improvement vs Ed

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.data_dir / "clean"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def golden_dir(self) -> Path:
        return self.data_dir / "golden"

    @property
    def vectorstore_path(self) -> Path:
        return self.data_dir / "vectorstore"


def get_settings() -> Settings:
    return Settings()
