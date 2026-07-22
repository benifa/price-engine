"""Build HuggingFace prompt/completion datasets from cleaned splits."""

from __future__ import annotations

import logging

from priceengine.config import Settings
from priceengine.corpus.io import load_listings, save_json
from priceengine.training.prompts import training_example, truncate_text

logger = logging.getLogger(__name__)


def build_sft_dataset(
    settings: Settings,
    *,
    style: str,
    cutoff: int,
    tokenizer_name: str = "meta-llama/Llama-3.2-3B",
    push_repo: str | None = None,
) -> dict[str, int]:
    """Read train/val/test parquet, truncate, emit prompt/completion, optional HF push."""
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    splits = {}
    counts = {}
    truncated_counts = {}
    for split_name in ("train", "val", "test"):
        path = settings.splits_dir / f"{split_name}.parquet"
        listings = load_listings(path)
        rows = []
        n_trunc = 0
        for listing in listings:
            text, was_trunc = truncate_text(listing.text_for_pricing(), tokenizer, cutoff)
            if was_trunc:
                n_trunc += 1
            rows.append(training_example(listing, style=style, text=text))
        splits[split_name] = Dataset.from_list(rows)
        counts[split_name] = len(rows)
        truncated_counts[split_name] = n_trunc
        logger.info(
            "%s: %d examples (%d truncated at cutoff=%d)",
            split_name,
            len(rows),
            n_trunc,
            cutoff,
        )

    # HF SFTTrainer often expects "train" / "validation"
    ds = DatasetDict(
        {
            "train": splits["train"],
            "validation": splits["val"],
            "test": splits["test"],
        }
    )
    out_dir = settings.data_dir / "hf_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    save_json(
        settings.reports_dir / "dataset_prep.json",
        {"style": style, "cutoff": cutoff, "counts": counts, "truncated": truncated_counts},
    )
    if push_repo:
        ds.push_to_hub(push_repo, private=True)
        logger.info("Pushed dataset to %s", push_repo)
    return counts
