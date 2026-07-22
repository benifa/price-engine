"""Evaluation harness — run contestants, write leaderboard reports."""

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
from priceengine.models import EvalItem, Prediction

logger = logging.getLogger(__name__)


def run_pricer(pricer: Pricer, items: list[EvalItem]) -> list[Prediction]:
    preds: list[Prediction] = []
    for i, item in enumerate(items):
        guess = float(pricer.price(item))
        error = abs(guess - item.price)
        preds.append(
            Prediction(
                item_id=item.id,
                guess=guess,
                truth=item.price,
                error=error,
                truncated=item.truncated,
                category=item.category,
                condition=item.condition,
            )
        )
        if (i + 1) % 50 == 0:
            logger.info("%s: %d / %d", pricer.name, i + 1, len(items))
    return preds


def write_leaderboard(
    *,
    battleground: str,
    results: dict[str, list[Prediction]],
    comparisons: list | None = None,
    settings: Settings | None = None,
    path: Path | None = None,
) -> Path:
    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    path = path or settings.reports_dir / f"leaderboard-{battleground}.md"

    lines = [
        f"# Leaderboard — `{battleground}`",
        "",
        "| Contestant | n | MAE | Median APE | Hit rate | RMSLE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    metrics = []
    for name, preds in results.items():
        m = summarize(name, battleground, preds, settings=settings)
        metrics.append(m)
        lines.append(format_metrics_row(m))

    lines.append("")
    if comparisons:
        lines.append("## Paired comparisons")
        lines.append("")
        for cmp_ in comparisons:
            lines.append(f"- {format_comparison(cmp_)}")
            lines.append("")

    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "- Hit = absolute error < $40 **or** relative error < 20% (Ed's definition)."
    )
    lines.append(
        "- Fine-tuned contestants are graded with their **training-native** prompt format."
    )
    lines.append(
        "- MAE CI is a 95% bootstrap over items; paired ΔMAE CI is a paired bootstrap."
    )
    lines.append("")

    path.write_text("\n".join(lines))
    # Also dump raw predictions for later analysis
    raw_path = path.with_suffix(".json")
    raw_path.write_text(
        json.dumps(
            {name: [p.model_dump() for p in preds] for name, preds in results.items()},
            indent=2,
        )
    )
    logger.info("Wrote %s and %s", path, raw_path)
    return path


def compare_to_baseline(
    ours_name: str,
    baseline_name: str,
    results: dict[str, list[Prediction]],
    *,
    battleground: str,
    settings: Settings | None = None,
):
    return paired_compare(
        ours_name,
        baseline_name,
        results[ours_name],
        results[baseline_name],
        battleground=battleground,
        settings=settings,
    )
