"""Score pricers on a golden set and write leaderboard markdown + JSON.

CLI entrypoints ``eval`` / ``eval-baselines`` call into this module. Model
adapters live in ``pricers`` / ``published_baseline``; math lives in ``metrics``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from priceengine.config import Settings, get_settings
from priceengine.eval.metrics import (
    format_comparison,
    format_metrics_row,
    paired_compare,
    summarize,
)
from priceengine.eval.pricers import Pricer
from priceengine.models import ComparisonResult, EvalItem, Prediction

logger = logging.getLogger(__name__)

_PROTOCOL_NOTES = (
    "- Hit = absolute error < $40 **or** relative error < 20%.",
    "- Fine-tuned models use the list-price prompt format (``Price is $``).",
    "- MAE CI is a 95% bootstrap over items; paired ΔMAE CI is a paired bootstrap.",
)


def run_pricer(pricer: Pricer, items: list[EvalItem]) -> list[Prediction]:
    """Score every golden item with one pricer; return per-item predictions."""
    predictions: list[Prediction] = []
    for index, item in enumerate(items):
        guess = float(pricer.price(item))
        predictions.append(
            Prediction(
                item_id=item.id,
                guess=guess,
                truth=item.price,
                error=abs(guess - item.price),
                truncated=item.truncated,
                category=item.category,
                condition=item.condition,
            )
        )
        if (index + 1) % 50 == 0:
            logger.info("%s: %d / %d", pricer.name, index + 1, len(items))
    return predictions


def write_leaderboard(
    *,
    eval_set: str,
    results: dict[str, list[Prediction]],
    comparisons: list[ComparisonResult] | None = None,
    settings: Settings | None = None,
    path: Path | None = None,
) -> Path:
    """Write ``leaderboard-{eval_set}.md`` and a twin ``.json`` of raw predictions.

    The markdown file is human-readable; the JSON is reused by the CLI to skip
    re-scoring the published baseline when item ids still match.
    """
    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    path = path or settings.reports_dir / f"leaderboard-{eval_set}.md"

    lines = [
        f"# Leaderboard — `{eval_set}`",
        "",
        "| Model | n | MAE | Median APE | Hit rate | RMSLE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, preds in results.items():
        metrics = summarize(name, eval_set, preds, settings=settings)
        lines.append(format_metrics_row(metrics))

    lines.append("")
    if comparisons:
        lines.append("## Paired comparisons")
        lines.append("")
        for comparison in comparisons:
            lines.append(f"- {format_comparison(comparison)}")
            lines.append("")

    lines.append("## Protocol")
    lines.append("")
    lines.extend(_PROTOCOL_NOTES)
    lines.append("")

    path.write_text("\n".join(lines))
    raw_path = path.with_suffix(".json")
    raw_path.write_text(
        json.dumps(
            {
                name: [prediction.model_dump() for prediction in preds]
                for name, preds in results.items()
            },
            indent=2,
        )
    )
    logger.info("Wrote %s and %s", path, raw_path)
    return path


def compare_to_baseline(
    challenger_name: str,
    baseline_name: str,
    results: dict[str, list[Prediction]],
    *,
    eval_set: str,
    settings: Settings | None = None,
) -> ComparisonResult:
    """Look up two named result lists and run ``paired_compare``."""
    return paired_compare(
        challenger_name,
        baseline_name,
        results[challenger_name],
        results[baseline_name],
        eval_set=eval_set,
        settings=settings,
    )
