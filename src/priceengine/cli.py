"""Typer CLI for list-price data prep, token budgets, SFT dataset build, and eval."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

app = typer.Typer(
    help="price-engine — Amazon list-price QLoRA research & fair eval",
    no_args_is_help=True,
)
log = logging.getLogger(__name__)


def _bootstrap():
    load_dotenv(override=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    from priceengine.config import get_settings

    return get_settings()


@app.command("prepare-list-prices")
def prepare_list_prices_cmd(
    size: str = typer.Option(
        "lite",
        help="lite = small official splits (default); full = large dataset subsample",
    ),
    train_limit: int = typer.Option(
        0,
        help="Cap train rows (0 = full lite split, or 100k default for full)",
    ),
    val_limit: int = typer.Option(2_000, help="Val cap when size=full"),
    dataset: str = typer.Option("", help="Optional HF dataset id override"),
):
    """Download Amazon list-price data from Hub → data/splits + golden."""
    settings = _bootstrap()
    from priceengine.data_prep import prepare_dataset

    kwargs: dict = {"size": size, "dataset_id": dataset or None}
    if size == "lite":
        kwargs["train_limit"] = train_limit or None
    else:
        kwargs["train_limit"] = train_limit or 100_000
        kwargs["val_limit"] = val_limit
    counts = prepare_dataset(settings, **kwargs)
    typer.echo(counts)


@app.command("token-budget")
def token_budget():
    """Write token-length histograms and CUTOFF table under reports/token_length/."""
    settings = _bootstrap()
    from priceengine.training.token_budget import analyze_token_budget

    result = analyze_token_budget(settings)
    typer.echo(json.dumps(result, indent=2))


@app.command("build-sft-dataset")
def build_sft_dataset_cmd(
    cutoff: int = typer.Option(110, help="Summary token CUTOFF"),
    push_repo: str = typer.Option("", help="Optional private HF dataset repo id"),
):
    """Build list-price prompt/completion HF dataset from local splits."""
    settings = _bootstrap()
    from priceengine.training.sft_dataset import build_sft_dataset

    counts = build_sft_dataset(
        settings, cutoff=cutoff, push_repo=push_repo or None
    )
    typer.echo(counts)


@app.command("eval-baselines")
def eval_baselines(
    golden: Path = typer.Option(
        Path("data/golden/amazon.parquet"), help="Golden-set parquet"
    ),
    train: Path = typer.Option(
        Path("data/splits/train.parquet"), help="Train parquet for category medians"
    ),
    limit: int = typer.Option(0, help="Optional cap on golden items (0 = all)"),
):
    """Grade dumb baselines on the golden set (no GPU)."""
    settings = _bootstrap()
    from priceengine.data_prep import load_eval_items, load_products, products_to_eval_items
    from priceengine.eval.leaderboard import run_pricer, write_leaderboard
    from priceengine.eval.pricers import OverallMedianPricer, SameCategoryMedianPricer

    items = load_eval_items(golden)
    if limit:
        items = items[:limit]
    train_items = products_to_eval_items(load_products(train))
    overall_median = sorted(i.price for i in train_items)[len(train_items) // 2]

    models = [
        SameCategoryMedianPricer(train_items),
        OverallMedianPricer(overall_median),
    ]
    results = {m.name: run_pricer(m, items) for m in models}
    path = write_leaderboard(
        eval_set="baselines",
        results=results,
        path=settings.reports_dir / "leaderboard-baselines.md",
    )
    typer.echo(f"Wrote {path}")


@app.command("eval")
def eval_cmd(
    golden: Path = typer.Option(
        Path("data/golden/amazon.parquet"), help="Golden-set parquet"
    ),
    eval_set: str = typer.Option(
        "",
        help="Leaderboard label (default: golden file stem)",
    ),
    limit: int = typer.Option(100, help="How many test items"),
    adapter_path: str = typer.Option(
        "",
        help="Challenger adapter (local dir, or Modal volume path with --modal). "
        "Empty = baselines + published baseline only.",
    ),
    name: str = typer.Option(
        "",
        help="Leaderboard label for the adapter (default: adapter path basename)",
    ),
    modal: bool = typer.Option(False, help="Score challenger adapter on Modal GPU"),
    include_baseline: bool = typer.Option(
        True,
        help="Include published Modal baseline (requires deployed pricer-service + HF_TOKEN)",
    ),
    baseline_preds: Path = typer.Option(
        Path("reports/leaderboard.json"),
        help="Reuse prior baseline predictions if item ids match",
    ),
    modal_app: str = typer.Option("pricer-service", help="Modal app for baseline"),
    out: Path = typer.Option(
        Path("reports/leaderboard.md"), help="Leaderboard markdown path"
    ),
):
    """Grade models on the Amazon golden set."""
    settings = _bootstrap()
    from priceengine.config import BASE_MODEL, SUMMARY_CUTOFF
    from priceengine.data_prep import load_eval_items, load_products, products_to_eval_items
    from priceengine.eval.leaderboard import compare_to_baseline, run_pricer, write_leaderboard
    from priceengine.eval.pricers import (
        OVERALL_TRAIN_MEDIAN_NAME,
        SAME_CATEGORY_MEDIAN_NAME,
        FineTunedPricer,
        OverallMedianPricer,
        SameCategoryMedianPricer,
    )
    from priceengine.eval.published_baseline import BASELINE_NAME, PublishedBaselinePricer
    from priceengine.models import Prediction
    from priceengine.training.prompts import prompt_for_eval_item, truncate_text

    set_name = eval_set.strip() or golden.stem
    _BASELINE_KEYS = (SAME_CATEGORY_MEDIAN_NAME, OVERALL_TRAIN_MEDIAN_NAME)

    items = load_eval_items(golden)
    if limit:
        items = items[:limit]
    train_path = settings.splits_dir / "train.parquet"
    if not train_path.exists():
        raise typer.BadParameter(f"Missing train split at {train_path}")
    train_items = products_to_eval_items(load_products(train_path))
    if not train_items:
        raise typer.BadParameter(f"Train split is empty: {train_path}")
    overall_median = sorted(i.price for i in train_items)[len(train_items) // 2]

    results: dict[str, list[Prediction]] = {
        SAME_CATEGORY_MEDIAN_NAME: run_pricer(
            SameCategoryMedianPricer(train_items), items
        ),
        OVERALL_TRAIN_MEDIAN_NAME: run_pricer(
            OverallMedianPricer(overall_median), items
        ),
    }
    comparisons = []

    if include_baseline:
        baseline_preds_list: list[Prediction] | None = None
        if baseline_preds.exists():
            raw = json.loads(baseline_preds.read_text())
            if BASELINE_NAME in raw:
                by_id = {p["item_id"]: p for p in raw[BASELINE_NAME]}
                if all(item.id in by_id for item in items):
                    baseline_preds_list = [
                        Prediction(**by_id[item.id]) for item in items
                    ]
                    typer.echo(f"Reused baseline preds from {baseline_preds}")
        if baseline_preds_list is None:
            typer.echo("Scoring published baseline on Modal…")
            baseline_preds_list = run_pricer(
                PublishedBaselinePricer(modal_app=modal_app), items
            )
        results[BASELINE_NAME] = baseline_preds_list

    if adapter_path:
        challenger_name = name.strip() or Path(adapter_path).name or "adapter"
        if modal:
            import modal as modal_sdk
            from transformers import AutoTokenizer

            from priceengine.eval import adapter_scoring as scoring

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            prompts = []
            for item in items:
                body, _ = truncate_text(
                    item.text_for_pricing(), tokenizer, SUMMARY_CUTOFF
                )
                prompts.append(prompt_for_eval_item(item, text=body))
            typer.echo(f"Scoring adapter on Modal ({adapter_path})…")
            with modal_sdk.enable_output():
                with scoring.app.run():
                    guesses = scoring.price_batch.remote(prompts, adapter_path)
            challenger_preds = [
                Prediction(
                    item_id=item.id,
                    guess=float(guess),
                    truth=item.price,
                    error=abs(float(guess) - item.price),
                    truncated=item.truncated,
                    category=item.category,
                    condition=item.condition,
                )
                for item, guess in zip(items, guesses, strict=True)
            ]
        else:
            typer.echo(f"Scoring adapter locally: {adapter_path}")
            challenger = FineTunedPricer(
                str(adapter_path),
                name=challenger_name,
                load_in_4bit=False,
            )
            challenger_preds = run_pricer(challenger, items)

        results[challenger_name] = challenger_preds
        if include_baseline and BASELINE_NAME in results:
            comparisons.append(
                compare_to_baseline(
                    challenger_name,
                    BASELINE_NAME,
                    results,
                    eval_set=set_name,
                )
            )

    ordered: dict[str, list[Prediction]] = {}
    for key in (BASELINE_NAME,):
        if key in results:
            ordered[key] = results.pop(key)
    for key, preds in results.items():
        if key not in _BASELINE_KEYS:
            ordered[key] = preds
    for key in _BASELINE_KEYS:
        if key in results:
            ordered[key] = results[key]

    path = write_leaderboard(
        eval_set=set_name,
        results=ordered,
        comparisons=comparisons or None,
        path=out,
    )
    typer.echo(f"Wrote {path}")


@app.command("visualize-eval")
def visualize_eval_cmd(
    leaderboard: Path = typer.Option(
        Path("reports/leaderboard.json"),
        help="Raw predictions from `priceengine eval`",
    ),
    golden: Path = typer.Option(
        Path("data/golden/amazon.parquet"), help="Golden set (for titles in hover)"
    ),
    out: Path = typer.Option(
        Path("reports/eval_report.html"), help="Self-contained HTML report path"
    ),
    eval_set: str = typer.Option("", help="Label in the report title"),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the HTML file in your browser"
    ),
):
    """Write course-style Plotly charts (truth vs guess + running MAE) to HTML."""
    settings = _bootstrap()
    from priceengine.eval.visualization import write_eval_html

    if not leaderboard.exists():
        raise typer.BadParameter(
            f"Missing {leaderboard}; run `priceengine eval` first."
        )
    path = write_eval_html(
        leaderboard,
        golden=golden,
        out=out,
        eval_set=eval_set.strip() or None,
        settings=settings,
        open_browser=open_browser,
    )
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
