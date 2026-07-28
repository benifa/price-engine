"""Build a HuggingFace prompt/completion dataset from local parquet splits.

Two data stages
---------------
1. ``data_prep.dataset_builder`` — Hub → ``data/splits/*.parquet`` (``ProductListing``)
2. **This module** — splits → ``data/hf_dataset/`` rows with ``prompt`` + ``completion``

When to use
-----------
The Modal replica job usually loads ``ed-donner/items_prompts_full`` from the Hub
(already prompt/completion shaped). Use this path when you want a local or
custom SFT set built from your own prepared splits.

CLI: ``priceengine build-sft-dataset --cutoff 110``.
"""

from __future__ import annotations

import logging

from priceengine.config import BASE_MODEL, Settings
from priceengine.data_prep import load_products, save_json
from priceengine.models import ProductListing
from priceengine.training.prompts import training_example, truncate_text

logger = logging.getLogger(__name__)

_SPLIT_NAMES = ("train", "val", "test")


def build_sft_dataset(
    settings: Settings,
    *,
    cutoff: int,
    tokenizer_name: str | None = None,
    push_repo: str | None = None,
) -> dict[str, int]:
    """Read train/val/test parquet, truncate, emit prompt/completion, optional HF push.

    Returns a dict of split → example count.
    """
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or BASE_MODEL)
    split_datasets = {}
    counts: dict[str, int] = {}
    truncated_counts: dict[str, int] = {}

    for split_name in _SPLIT_NAMES:
        products = load_products(settings.splits_dir / f"{split_name}.parquet")
        rows, n_trunc = _examples_from_products(products, tokenizer, cutoff)
        split_datasets[split_name] = Dataset.from_list(rows)
        counts[split_name] = len(rows)
        truncated_counts[split_name] = n_trunc
        logger.info(
            "%s: %d examples (%d truncated at cutoff=%d)",
            split_name,
            len(rows),
            n_trunc,
            cutoff,
        )

    dataset = DatasetDict(
        {
            "train": split_datasets["train"],
            "validation": split_datasets["val"],
            "test": split_datasets["test"],
        }
    )
    out_dir = settings.data_dir / "hf_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir))
    save_json(
        settings.reports_dir / "sft_dataset.json",
        {
            "format": "amazon_list",
            "cutoff": cutoff,
            "counts": counts,
            "truncated": truncated_counts,
        },
    )
    if push_repo:
        dataset.push_to_hub(push_repo, private=True)
        logger.info("Pushed dataset to %s", push_repo)
    return counts


def _examples_from_products(
    products: list[ProductListing],
    tokenizer,
    cutoff: int,
) -> tuple[list[dict[str, str]], int]:
    """Convert listings to SFT rows; return ``(rows, n_truncated)``."""
    rows: list[dict[str, str]] = []
    n_truncated = 0
    for product in products:
        text, was_truncated = truncate_text(
            product.text_for_pricing(), tokenizer, cutoff
        )
        if was_truncated:
            n_truncated += 1
        rows.append(training_example(product, text=text))
    return rows, n_truncated
