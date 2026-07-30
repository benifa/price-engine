"""Build prompt/completion SFT rows from local item splits.

Reference path (preferred when available)
-----------------------------------------
Modal training loads ``benifa/items_prompts_full`` from the Hub — already shaped
as ``prompt`` + ``completion``. That avoids rebuilding 800k rows locally.

Fallback (save Hub download / offline)
--------------------------------------
When the Hub prompts dataset is missing or ``dataset.hub_id`` is null:

1. ``priceengine prepare-data`` → ``data/splits/*.parquet`` (raw items)
2. ``priceengine build-local-sft`` → ``data/hf_dataset/`` (this module)
3. Copy/sync ``hf_dataset`` onto the Modal volume at ``/data/hf_dataset``
4. Train with ``dataset.hub_id: null`` (or let Modal fall back after Hub miss)

Each row is generated the same way Ed’s Hub set was::

    prompt     = question + product text + \"Price is $\"
    completion = \"NNN.00\"
"""

from __future__ import annotations

import logging

from priceengine.config import BASE_MODEL, MAX_DESCRIPTION_TOKENS, Settings
from priceengine.data import load_products, save_json
from priceengine.models import ProductListing
from priceengine.prompts import training_example, truncate_text

logger = logging.getLogger(__name__)

_SPLIT_NAMES = ("train", "val", "test")


def build_local_sft(
    settings: Settings,
    *,
    cutoff: int = MAX_DESCRIPTION_TOKENS,
    tokenizer_name: str | None = None,
    push_repo: str | None = None,
) -> dict[str, int]:
    """Read split parquet → truncate → prompt/completion DatasetDict on disk.

    Writes ``data/hf_dataset/`` and ``reports/sft_dataset.json``.
    Returns split → row count.
    """
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or BASE_MODEL)
    split_datasets: dict[str, Dataset] = {}
    counts: dict[str, int] = {}
    truncated_counts: dict[str, int] = {}

    for split_name in _SPLIT_NAMES:
        path = settings.splits_dir / f"{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `priceengine prepare-data` first."
            )
        products = load_products(path)
        rows, n_truncated = prompt_completion_rows_from_products(
            products, tokenizer, cutoff
        )
        split_datasets[split_name] = Dataset.from_list(rows)
        counts[split_name] = len(rows)
        truncated_counts[split_name] = n_truncated
        logger.info(
            "%s: %d examples (%d truncated at cutoff=%d)",
            split_name,
            len(rows),
            n_truncated,
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
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        settings.reports_dir / "sft_dataset.json",
        {
            "format": "list_price",
            "source": "local_splits",
            "cutoff": cutoff,
            "counts": counts,
            "truncated": truncated_counts,
            "out_dir": str(out_dir),
        },
    )
    if push_repo:
        dataset.push_to_hub(push_repo, private=True)
        logger.info("Pushed dataset to %s", push_repo)
    return counts


def prompt_completion_rows_from_products(
    products: list[ProductListing],
    tokenizer,
    cutoff: int,
) -> tuple[list[dict[str, str]], int]:
    """Convert listings → SFT rows. Returns ``(rows, n_truncated)``.

    Uses ``description`` (Hub ``summary``) as the prompt body so local rows match
    ``items_prompts_full``. Falls back to title+description if summary is empty.
    """
    prompt_completion_rows: list[dict[str, str]] = []
    n_truncated = 0
    for product in products:
        body = (product.description or "").strip() or product.text_for_pricing()
        text, was_truncated = truncate_text(body, tokenizer, cutoff)
        if was_truncated:
            n_truncated += 1
        prompt_completion_rows.append(training_example(product, text=text))
    return prompt_completion_rows, n_truncated
