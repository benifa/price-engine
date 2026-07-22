"""Token-length histograms and CUTOFF selection reports."""

from __future__ import annotations

import logging

from priceengine.config import Settings
from priceengine.corpus.io import load_listings
from priceengine.training.prompts import ours_prompt, truncate_text

logger = logging.getLogger(__name__)


def analyze_token_budget(
    settings: Settings,
    *,
    tokenizer_name: str = "meta-llama/Llama-3.2-3B",
    cutoffs: list[int] | None = None,
    sample_size: int = 5000,
) -> dict:
    """Write histograms + truncation table for candidate CUTOFFs."""
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    cutoffs = cutoffs or [80, 100, 110, 130, 150]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    listings = load_listings(settings.splits_dir / "train.parquet")
    if len(listings) > sample_size:
        listings = listings[:sample_size]

    desc_counts = []
    full_counts = []
    for listing in listings:
        text = listing.text_for_pricing()
        desc_counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
        prompt = ours_prompt(
            text,
            condition=listing.condition,
            category=listing.category,
            sold_ym=listing.sold_date.strftime("%Y-%m"),
        )
        full = prompt + f"{round(listing.sold_price)}.00"
        full_counts.append(len(tokenizer.encode(full, add_special_tokens=False)))

    out_dir = settings.reports_dir / "token_length"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _hist(values: list[int], title: str, filename: str) -> None:
        plt.figure(figsize=(10, 4))
        plt.hist(values, bins=range(0, max(values + [1]) + 10, 10), color="steelblue", rwidth=0.85)
        avg = sum(values) / len(values)
        plt.title(f"{title} — avg {avg:.1f}, max {max(values)}")
        plt.xlabel("tokens")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()

    _hist(desc_counts, "Description tokens", "desc_tokens.png")
    _hist(full_counts, "Prompt + completion tokens", "full_tokens.png")

    rows = []
    for cutoff in cutoffs:
        n_trunc = sum(
            1
            for listing in listings
            if truncate_text(listing.text_for_pricing(), tokenizer, cutoff)[1]
        )
        rows.append(
            {
                "cutoff": cutoff,
                "truncation_rate": n_trunc / len(listings),
                "n": len(listings),
            }
        )

    report = out_dir / "cutoff_table.md"
    lines = [
        "# Token budget / CUTOFF",
        "",
        f"Sample size: {len(listings)}",
        "",
        "| CUTOFF | Truncation rate |",
        "|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['cutoff']} | {row['truncation_rate']:.1%} |")
    lines.append("")
    lines.append(
        "Recommended starting CUTOFF: **110** (Ed's day-2 choice) unless P95 description "
        "length clearly exceeds it — see histograms."
    )
    report.write_text("\n".join(lines))
    logger.info("Wrote token budget reports to %s", out_dir)
    return {"cutoffs": rows, "out_dir": str(out_dir)}
