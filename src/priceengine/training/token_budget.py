"""Token-length histograms and CUTOFF selection reports.

Purpose
-------
Choose how many *description* tokens to keep (``SUMMARY_CUTOFF``, default 110)
so that ``list_price_prompt(text) + completion`` still fits the training
``max_seq_length`` (128 in the Colab replica).

Writes under ``reports/token_length/``:

* ``desc_tokens.png`` — raw description lengths
* ``full_tokens.png`` — prompt + completion lengths
* ``cutoff_table.md`` — truncation rate at candidate CUTOFFs

CLI: ``priceengine token-budget`` (needs ``data/splits/train.parquet``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from priceengine.config import BASE_MODEL, Settings
from priceengine.data_prep import load_products
from priceengine.training.prompts import (
    list_price_prompt,
    price_completion,
    truncate_text,
)

logger = logging.getLogger(__name__)

_DEFAULT_CUTOFFS = [80, 100, 110, 130, 150]


def analyze_token_budget(
    settings: Settings,
    *,
    tokenizer_name: str | None = None,
    cutoffs: list[int] | None = None,
    sample_size: int = 5000,
) -> dict:
    """Sample the train split, write histograms + cutoff table, return summary dict."""
    from transformers import AutoTokenizer

    cutoffs = cutoffs or list(_DEFAULT_CUTOFFS)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or BASE_MODEL)
    products = load_products(settings.splits_dir / "train.parquet")
    if len(products) > sample_size:
        products = products[:sample_size]

    desc_counts: list[int] = []
    full_counts: list[int] = []
    for product in products:
        text = product.text_for_pricing()
        desc_counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
        full = list_price_prompt(text) + price_completion(product.list_price)
        full_counts.append(len(tokenizer.encode(full, add_special_tokens=False)))

    out_dir = settings.reports_dir / "token_length"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_histogram(out_dir / "desc_tokens.png", desc_counts, "Description tokens")
    _write_histogram(
        out_dir / "full_tokens.png",
        full_counts,
        "Prompt + completion (amazon_list)",
    )

    rows = [
        {
            "cutoff": cutoff,
            "truncation_rate": _truncation_rate(products, tokenizer, cutoff),
            "n": len(products),
        }
        for cutoff in cutoffs
    ]
    _write_cutoff_table(out_dir / "cutoff_table.md", rows, sample_size=len(products))
    logger.info("Wrote token budget reports to %s", out_dir)
    return {"format": "amazon_list", "cutoffs": rows, "out_dir": str(out_dir)}


def _truncation_rate(products, tokenizer, cutoff: int) -> float:
    n_trunc = sum(
        1
        for product in products
        if truncate_text(product.text_for_pricing(), tokenizer, cutoff)[1]
    )
    return n_trunc / len(products) if products else 0.0


def _write_histogram(path: Path, values: list[int], title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.hist(
        values,
        bins=range(0, max(values + [1]) + 10, 10),
        color="steelblue",
        rwidth=0.85,
    )
    avg = sum(values) / len(values) if values else 0.0
    plt.title(f"{title} — avg {avg:.1f}, max {max(values) if values else 0}")
    plt.xlabel("tokens")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_cutoff_table(
    path: Path, rows: list[dict], *, sample_size: int
) -> None:
    lines = [
        "# Token budget / CUTOFF",
        "",
        f"Format: `amazon_list` · sample size: {sample_size}",
        "",
        "| CUTOFF | Truncation rate |",
        "|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['cutoff']} | {row['truncation_rate']:.1%} |")
    lines.append("")
    lines.append(
        "Recommended starting CUTOFF: **110** unless P95 description length clearly "
        "exceeds it — see histograms."
    )
    path.write_text("\n".join(lines))
