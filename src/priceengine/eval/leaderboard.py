"""Score pricers and write leaderboard markdown + JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from priceengine.config import Settings, get_settings
from priceengine.eval.metrics import (
    format_comparison,
    paired_compare,
    summarize,
)
from priceengine.eval.pricers import Pricer
from priceengine.eval.roles import role_for, short_label
from priceengine.models import ComparisonResult, EvalItem, Prediction

logger = logging.getLogger(__name__)

_PROTOCOL_NOTES_TEMPLATE = (
    "- Hit = absolute error < ${hit_abs:,.0f} **or** relative error < {hit_rel:.0%}.",
    "- Fine-tuned models use the list-price prompt (``Price is $``).",
    "- MAE CI is a 95% bootstrap; paired ΔMAE CI is a paired bootstrap.",
)


def run_pricer(pricer: Pricer, items: list[EvalItem]) -> list[Prediction]:
    """Score every golden item with one pricer."""
    predictions: list[Prediction] = []
    for index, item in enumerate(items):
        estimate = float(pricer.price(item))
        predictions.append(
            Prediction(
                item_id=item.item_id,
                estimate=estimate,
                actual=item.price,
                error=abs(estimate - item.price),
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
    """Write leaderboard markdown and a twin JSON of raw predictions."""
    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    path = path or settings.reports_dir / f"leaderboard-{eval_set}.md"

    lines = [
        f"# Leaderboard — `{eval_set}`",
        "",
        "| Rank | Role | Model | n | MAE | Median APE | Hit rate | RMSLE |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        results.items(),
        key=lambda kv: summarize(
            kv[0], eval_set, kv[1], settings=settings, bootstrap=False
        ).mae,
    )
    for rank, (name, preds) in enumerate(ranked, start=1):
        metrics = summarize(name, eval_set, preds, settings=settings)
        ci = ""
        if metrics.mae_ci_low is not None and metrics.mae_ci_high is not None:
            ci = (
                f"  (95% CI ${metrics.mae_ci_low:,.2f}–"
                f"${metrics.mae_ci_high:,.2f})"
            )
        lines.append(
            f"| {rank} | {role_for(name)} | {short_label(name)} | {metrics.n} | "
            f"${metrics.mae:,.2f}{ci} | {metrics.median_ape:.1%} | "
            f"{metrics.hit_rate:.1%} | {metrics.rmsle:.3f} |"
        )

    lines.append("")
    if comparisons:
        lines.append("## Paired comparisons")
        lines.append("")
        for comparison in comparisons:
            lines.append(f"- {format_comparison(comparison)}")
            lines.append("")

    lines.append("## Protocol")
    lines.append("")
    lines.extend(
        note.format(
            hit_abs=settings.hit_abs_dollars,
            hit_rel=settings.hit_rel_fraction,
        )
        for note in _PROTOCOL_NOTES_TEMPLATE
    )
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
    return paired_compare(
        challenger_name,
        baseline_name,
        results[challenger_name],
        results[baseline_name],
        eval_set=eval_set,
        settings=settings,
    )
