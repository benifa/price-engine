"""Eval orchestration used by the CLI.

Flow (match this when reading the file)::

    run_eval
      1. load_golden_items
      2. score_median_baselines          # sanity floor (no GPU)
      3. load_or_score_published_baseline  # optional; may reuse JSON cache
      4. score_frontier_models           # optional --frontier gpt-5
      5. score_challenger_adapter        # optional --adapter-path
      6. compare_to_baseline             # only if baseline + challenger both ran
      7. write_leaderboard               # reports/leaderboard.md + .json
      8. write_eval_html                 # optional --visualize

Plain-language goal: on the **same** held-out products, ask each pricer “what
does this cost?”, then compare dollar errors. See ``docs/EVAL.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from priceengine.config import BASE_MODEL, MAX_DESCRIPTION_TOKENS, Settings
from priceengine.data import load_eval_items, load_products, products_to_eval_items
from priceengine.eval.baseline_pricer import BASELINE_NAME, PublishedBaselinePricer
from priceengine.eval.frontier_pricer import FrontierChatPricer, frontier_display_name
from priceengine.eval.leaderboard import compare_to_baseline, run_pricer, write_leaderboard
from priceengine.eval.pricers import (
    OVERALL_TRAIN_MEDIAN_NAME,
    SAME_CATEGORY_MEDIAN_NAME,
    FineTunedPricer,
    OverallMedianPricer,
    SameCategoryMedianPricer,
)
from priceengine.eval.roles import is_frontier, is_naive_floor
from priceengine.models import EvalItem, Prediction
from priceengine.prompts import prompt_for_eval_item, truncate_text

logger = logging.getLogger(__name__)

_MEDIAN_BASELINE_NAMES = (SAME_CATEGORY_MEDIAN_NAME, OVERALL_TRAIN_MEDIAN_NAME)


def load_golden_items(golden: Path, *, limit: int) -> list[EvalItem]:
    """Step 1 — held-out products we score everyone on (never used for training weights)."""
    items = load_eval_items(golden)
    if limit:
        items = items[:limit]
    return items


def score_median_baselines(
    items: list[EvalItem],
    *,
    train_parquet: Path,
) -> dict[str, list[Prediction]]:
    """Step 2 — dumb baselines from train prices (must beat these before claiming wins)."""
    if not train_parquet.exists():
        raise FileNotFoundError(f"Missing train split at {train_parquet}")
    train_items = products_to_eval_items(load_products(train_parquet))
    if not train_items:
        raise ValueError(f"Train split is empty: {train_parquet}")

    overall_median = sorted(item.price for item in train_items)[len(train_items) // 2]
    return {
        SAME_CATEGORY_MEDIAN_NAME: run_pricer(
            SameCategoryMedianPricer(train_items), items
        ),
        OVERALL_TRAIN_MEDIAN_NAME: run_pricer(
            OverallMedianPricer(overall_median), items
        ),
    }


def load_or_score_published_baseline(
    items: list[EvalItem],
    *,
    baseline_preds_path: Path,
    modal_app: str,
) -> list[Prediction]:
    """Step 3 — published Hub adapter via Modal, or reuse cached predictions.

    Cache hit requires every current golden ``item_id`` to appear under
    ``BASELINE_NAME`` in the JSON file (usually a prior ``leaderboard.json``).
    """
    if baseline_preds_path.exists():
        raw = json.loads(baseline_preds_path.read_text())
        if BASELINE_NAME in raw:
            by_item_id = {row["item_id"]: row for row in raw[BASELINE_NAME]}
            if all(item.item_id in by_item_id for item in items):
                logger.info("Reused baseline preds from %s", baseline_preds_path)
                return [
                    Prediction(**by_item_id[item.item_id]) for item in items
                ]

    logger.info("Scoring published baseline on Modal…")
    return run_pricer(PublishedBaselinePricer(modal_app=modal_app), items)


def score_frontier_models(
    items: list[EvalItem],
    *,
    models: list[str],
    preds_path: Path,
) -> dict[str, list[Prediction]]:
    """Step 4 — optional OpenAI frontier models (reuse cache when item ids match)."""
    cached: dict = {}
    if preds_path.exists():
        cached = json.loads(preds_path.read_text())

    out: dict[str, list[Prediction]] = {}
    for model in models:
        model = model.strip()
        if not model:
            continue
        name = frontier_display_name(model)
        if name in cached:
            by_item_id = {row["item_id"]: row for row in cached[name]}
            if all(item.item_id in by_item_id for item in items):
                logger.info("Reused frontier preds for %s from %s", name, preds_path)
                out[name] = [Prediction(**by_item_id[item.item_id]) for item in items]
                continue
        logger.info("Scoring frontier model %s…", model)
        out[name] = run_pricer(FrontierChatPricer(model), items)
    return out


def predictions_from_estimates(
    items: list[EvalItem],
    estimates: list[float],
) -> list[Prediction]:
    """Zip golden items with model dollar guesses into ``Prediction`` rows."""
    return [
        Prediction(
            item_id=item.item_id,
            estimate=float(estimate),
            actual=item.price,
            error=abs(float(estimate) - item.price),
            truncated=item.truncated,
            category=item.category,
            condition=item.condition,
        )
        for item, estimate in zip(items, estimates, strict=True)
    ]


def score_challenger_on_modal(
    items: list[EvalItem],
    *,
    adapter_path: str,
) -> list[Prediction]:
    """Score our adapter on Modal GPU (fair path: 4-bit, same prompt contract)."""
    import modal as modal_sdk
    from transformers import AutoTokenizer

    from priceengine.eval import modal_score as scoring

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    prompts: list[str] = []
    for item in items:
        body, _ = truncate_text(
            item.text_for_pricing(), tokenizer, MAX_DESCRIPTION_TOKENS
        )
        prompts.append(prompt_for_eval_item(item, text=body))

    logger.info("Scoring adapter on Modal (%s)…", adapter_path)
    with modal_sdk.enable_output():
        with scoring.app.run():
            estimates = scoring.score_prompts_with_adapter.remote(
                prompts, adapter_path
            )
    return predictions_from_estimates(items, estimates)


def score_challenger_locally(
    items: list[EvalItem],
    *,
    adapter_path: str,
    challenger_name: str,
) -> list[Prediction]:
    """Score adapter on this machine.

    Honest note: the CLI uses ``load_in_4bit=False`` here (friendlier on macOS/MPS).
    That is fine for smoke checks; for a fair claim vs the published baseline, prefer
    ``--modal`` so serve settings match the fair path in ``docs/EVAL.md``.
    """
    logger.info("Scoring adapter locally: %s", adapter_path)
    challenger = FineTunedPricer(
        str(adapter_path),
        name=challenger_name,
        load_in_4bit=False,
    )
    return run_pricer(challenger, items)


def score_challenger_adapter(
    items: list[EvalItem],
    *,
    adapter_path: str,
    challenger_name: str,
    use_modal: bool,
) -> list[Prediction]:
    """Step 5 — our fine-tuned adapter (Modal T4 batch or local load)."""
    if use_modal:
        return score_challenger_on_modal(items, adapter_path=adapter_path)
    return score_challenger_locally(
        items, adapter_path=adapter_path, challenger_name=challenger_name
    )


def order_leaderboard_rows(
    results: dict[str, list[Prediction]],
) -> dict[str, list[Prediction]]:
    """Stable table order: our model → frontier → specialist → naive floors."""
    ordered: dict[str, list[Prediction]] = {}
    remaining = dict(results)

    ours = [
        name
        for name in remaining
        if not is_naive_floor(name)
        and not is_frontier(name)
        and name != BASELINE_NAME
    ]
    frontiers = [name for name in remaining if is_frontier(name)]
    for name in ours + frontiers:
        ordered[name] = remaining.pop(name)
    if BASELINE_NAME in remaining:
        ordered[BASELINE_NAME] = remaining.pop(BASELINE_NAME)
    for name in _MEDIAN_BASELINE_NAMES:
        if name in remaining:
            ordered[name] = remaining.pop(name)
    ordered.update(remaining)
    return ordered


def default_eval_set_label(golden: Path, eval_set: str) -> str:
    """Prefer a readable label over the parquet stem ``amazon``."""
    if eval_set.strip():
        return eval_set.strip()
    if golden.stem == "amazon":
        return "list-price golden"
    return golden.stem


def run_eval(
    settings: Settings,
    *,
    golden: Path,
    eval_set: str = "",
    limit: int = 100,
    adapter_path: str = "",
    name: str = "",
    modal: bool = False,
    include_baseline: bool = True,
    baseline_preds: Path = Path("reports/leaderboard.json"),
    frontier: list[str] | None = None,
    modal_app: str = "pricer-service",
    out: Path = Path("reports/leaderboard.md"),
    visualize: bool = False,
    report_version: str = "",
    open_browser: bool = False,
) -> Path:
    """Run the full eval pipeline; return path to the markdown leaderboard."""
    eval_set_name = default_eval_set_label(golden, eval_set)

    # 1
    golden_items = load_golden_items(golden, limit=limit)
    # 2
    results = score_median_baselines(
        golden_items, train_parquet=settings.splits_dir / "train.parquet"
    )
    comparisons = []

    # 3
    if include_baseline:
        results[BASELINE_NAME] = load_or_score_published_baseline(
            golden_items,
            baseline_preds_path=baseline_preds,
            modal_app=modal_app,
        )

    # 4
    if frontier:
        results.update(
            score_frontier_models(
                golden_items,
                models=frontier,
                preds_path=baseline_preds,
            )
        )

    # 5 (+ 6 if both sides present)
    if adapter_path:
        challenger_name = name.strip() or Path(adapter_path).name or "adapter"
        results[challenger_name] = score_challenger_adapter(
            golden_items,
            adapter_path=adapter_path,
            challenger_name=challenger_name,
            use_modal=modal,
        )
        if include_baseline and BASELINE_NAME in results:
            comparisons.append(
                compare_to_baseline(
                    challenger_name,
                    BASELINE_NAME,
                    results,
                    eval_set=eval_set_name,
                )
            )
        for frontier_name in list(results):
            if is_frontier(frontier_name):
                comparisons.append(
                    compare_to_baseline(
                        challenger_name,
                        frontier_name,
                        results,
                        eval_set=eval_set_name,
                    )
                )

    # 7
    leaderboard_path = write_leaderboard(
        eval_set=eval_set_name,
        results=order_leaderboard_rows(results),
        comparisons=comparisons or None,
        path=out,
        settings=settings,
    )

    # 8
    if visualize:
        from priceengine.eval.report_html import write_eval_html

        html_path = write_eval_html(
            leaderboard_path.with_suffix(".json"),
            golden=golden,
            out=settings.reports_dir / "eval_report.html",
            eval_set=eval_set_name,
            settings=settings,
            open_browser=open_browser,
            version=report_version.strip() or None,
        )
        logger.info("Wrote %s", html_path)

    return leaderboard_path
