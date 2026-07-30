"""CLI — prepare → (optional local SFT) → train (Modal) → eval → publish.

    priceengine prepare-data --size lite
    priceengine build-local-sft   # only if Hub prompts unavailable
    modal run src/priceengine/train/modal_train.py --config configs/qlora.yaml
    priceengine eval --modal --adapter-path /data/checkpoints/list_price_qlora --visualize
    priceengine publish-model --adapter-path … --tag v0.1.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from priceengine.config import MAX_DESCRIPTION_TOKENS, get_settings

app = typer.Typer(
    help="price-engine — Amazon list-price QLoRA research & fair eval",
    no_args_is_help=True,
)


def _bootstrap():
    load_dotenv(override=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    return get_settings()


@app.command("prepare-data")
def prepare_data_cmd(
    size: str = typer.Option(
        "lite",
        help="lite = small official splits; full = large subsample",
    ),
    train_limit: int = typer.Option(
        0,
        help="Cap train rows (0 = full lite split, or 100k default for full)",
    ),
    val_limit: int = typer.Option(2_000, help="Val cap when size=full"),
    dataset: str = typer.Option("", help="Optional HF dataset id override"),
):
    """Download Hub data → data/splits + data/golden."""
    settings = _bootstrap()
    from priceengine.data import prepare_dataset

    kwargs: dict = {"size": size, "dataset_id": dataset or None}
    if size == "lite":
        kwargs["train_limit"] = train_limit or None
    else:
        kwargs["train_limit"] = train_limit or 100_000
        kwargs["val_limit"] = val_limit
    typer.echo(prepare_dataset(settings, **kwargs))


@app.command("build-local-sft")
def build_local_sft_cmd(
    cutoff: int = typer.Option(
        MAX_DESCRIPTION_TOKENS,
        help="Max description tokens (config MAX_DESCRIPTION_TOKENS)",
    ),
    push_repo: str = typer.Option(
        "",
        help="Optional: push DatasetDict to a private Hub repo id",
    ),
):
    """Build prompt/completion dataset from data/splits (Hub-prompts fallback)."""
    settings = _bootstrap()
    from priceengine.train.local_sft import build_local_sft

    try:
        counts = build_local_sft(
            settings, cutoff=cutoff, push_repo=push_repo or None
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(counts)


@app.command("eval")
def eval_cmd(
    golden: Path = typer.Option(
        Path("data/golden/amazon.parquet"), help="Golden-set parquet"
    ),
    eval_set: str = typer.Option("", help="Leaderboard label (default: file stem)"),
    limit: int = typer.Option(100, help="How many test items"),
    adapter_path: str = typer.Option(
        "",
        help="Challenger adapter path (Modal volume path with --modal). "
        "Empty = medians + published baseline only.",
    ),
    name: str = typer.Option("", help="Leaderboard label for the adapter"),
    modal: bool = typer.Option(False, help="Score challenger on Modal GPU"),
    include_baseline: bool = typer.Option(
        True, help="Include published Modal baseline"
    ),
    baseline_preds: Path = typer.Option(
        Path("reports/leaderboard.json"),
        help="Reuse prior baseline/frontier predictions if item ids match",
    ),
    frontier: list[str] = typer.Option(
        [],
        "--frontier",
        help="OpenAI model id(s) to score, e.g. --frontier gpt-5 "
        "(repeatable; needs OPENAI_API_KEY + uv sync --extra frontier)",
    ),
    modal_app: str = typer.Option("pricer-service", help="Modal app for baseline"),
    out: Path = typer.Option(
        Path("reports/leaderboard.md"), help="Leaderboard markdown path"
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize/--no-visualize",
        help="Also write reports/eval_report.html",
    ),
    report_version: str = typer.Option(
        "", help="Optional HTML version tag (e.g. v0.1.0)"
    ),
    open_browser: bool = typer.Option(
        False, "--open/--no-open", help="Open HTML when --visualize is set"
    ),
):
    """Score naive floors + optional frontier / baseline / challenger on the golden set."""
    settings = _bootstrap()
    from priceengine.eval.run import run_eval

    try:
        path = run_eval(
            settings,
            golden=golden,
            eval_set=eval_set,
            limit=limit,
            adapter_path=adapter_path,
            name=name,
            modal=modal,
            include_baseline=include_baseline,
            baseline_preds=baseline_preds,
            frontier=frontier or None,
            modal_app=modal_app,
            out=out,
            visualize=visualize,
            report_version=report_version,
            open_browser=open_browser,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote {path}")


@app.command("publish-model")
def publish_model_cmd(
    adapter_path: Path = typer.Option(
        ...,
        "--adapter-path",
        help="Local LoRA adapter directory",
    ),
    repo: str = typer.Option("benifa/list-price-qlora", help="HF model repo id"),
    tag: str = typer.Option(..., "--tag", help="Revision tag (v0.1.0)"),
    public: bool = typer.Option(
        False, "--public/--private", help="Hub visibility (default: private)"
    ),
    leaderboard_md: Path = typer.Option(
        Path("reports/leaderboard.md"),
        help="Metrics snapshot for the model card",
    ),
):
    """Upload a versioned LoRA adapter to the Hugging Face Hub."""
    settings = _bootstrap()
    from priceengine.train.publish import publish_adapter

    result = publish_adapter(
        adapter_path,
        repo_id=repo,
        tag=tag,
        private=not public,
        leaderboard_md=leaderboard_md if leaderboard_md.exists() else None,
        settings=settings,
    )
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
