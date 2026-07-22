"""Typer CLI for corpus prep, token budgets, dataset build, and local eval helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

app = typer.Typer(
    help="price-engine — sold-price valuation research & serving",
    no_args_is_help=True,
)
log = logging.getLogger(__name__)


def _bootstrap():
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from priceengine.config import get_settings

    return get_settings()


@app.command("prepare-items-lite")
def prepare_items_lite_cmd(
    train_limit: int = typer.Option(
        0, help="Optional cap on train rows (0 = full 20k lite split)"
    ),
    dataset: str = typer.Option("ed-donner/items_lite", help="HF dataset id"),
):
    """Download Ed's items_lite and write splits + golden set (no Apify)."""
    settings = _bootstrap()
    from priceengine.corpus.items_lite import prepare_items_lite

    counts = prepare_items_lite(
        settings,
        dataset_name=dataset,
        train_limit=train_limit or None,
    )
    typer.echo(counts)


@app.command("pull-apify")
def pull_apify(
    max_items: int = typer.Option(50_000, help="Max items across all queries"),
    actor: str = typer.Option("dtrungtin/ebay-items-scraper", help="Apify actor id"),
):
    """Pull eBay sold listings via Apify and write data/raw/*.jsonl."""
    settings = _bootstrap()
    from priceengine.corpus.apify_pull import DEFAULT_QUERIES, pull_sold_listings
    from priceengine.corpus.cleaning import clean_listings, drop_log_outliers
    from priceengine.corpus.io import save_json, save_listings

    raw = pull_sold_listings(
        settings, queries=DEFAULT_QUERIES, max_items=max_items, actor_id=actor
    )
    cleaned, drops = clean_listings(raw)
    cleaned, n_out = drop_log_outliers(cleaned)
    drops["log_outlier"] = n_out
    settings.clean_dir.mkdir(parents=True, exist_ok=True)
    save_listings(settings.clean_dir / "sold.parquet", cleaned)
    save_json(settings.reports_dir / "cleaning_report.json", {"kept": len(cleaned), "drops": drops})
    typer.echo(f"Kept {len(cleaned)} listings. Drops: {drops}")


@app.command("prepare-from-jsonl")
def prepare_from_jsonl(
    path: Path = typer.Argument(..., help="Path to an Apify JSONL dump"),
):
    """Clean + split a local JSONL dump (no Apify call)."""
    settings = _bootstrap()
    from priceengine.corpus.apify_pull import load_raw_jsonl
    from priceengine.corpus.cleaning import clean_listings, drop_log_outliers
    from priceengine.corpus.io import save_eval_items, save_json, save_listings
    from priceengine.corpus.splits import (
        remove_boundary_near_dupes,
        stratify_report,
        time_split,
        to_eval_items,
    )

    raw = load_raw_jsonl(path)
    cleaned, drops = clean_listings(raw)
    cleaned, n_out = drop_log_outliers(cleaned)
    drops["log_outlier"] = n_out
    save_listings(settings.clean_dir / "sold.parquet", cleaned)

    train, val, test = time_split(cleaned)
    train, val, test, n_dupe = remove_boundary_near_dupes(train, val, test)
    drops["boundary_near_dupes"] = n_dupe
    settings.splits_dir.mkdir(parents=True, exist_ok=True)
    save_listings(settings.splits_dir / "train.parquet", train)
    save_listings(settings.splits_dir / "val.parquet", val)
    save_listings(settings.splits_dir / "test.parquet", test)

    golden = to_eval_items(test)
    save_eval_items(settings.golden_dir / "used_goods.parquet", golden)
    save_json(
        settings.reports_dir / "split_report.json",
        {
            "kept": len(cleaned),
            "drops": drops,
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "train_strata": stratify_report(train),
            "test_strata": stratify_report(test),
        },
    )
    typer.echo(
        f"Prepared splits: train={len(train)} val={len(val)} test={len(test)} "
        f"(golden={len(golden)})"
    )


@app.command("split-clean")
def split_clean():
    """Time-split data/clean/sold.parquet → splits + golden set."""
    settings = _bootstrap()
    from priceengine.corpus.io import load_listings, save_eval_items, save_json, save_listings
    from priceengine.corpus.splits import (
        remove_boundary_near_dupes,
        stratify_report,
        time_split,
        to_eval_items,
    )

    cleaned = load_listings(settings.clean_dir / "sold.parquet")
    train, val, test = time_split(cleaned)
    train, val, test, n_dupe = remove_boundary_near_dupes(train, val, test)
    save_listings(settings.splits_dir / "train.parquet", train)
    save_listings(settings.splits_dir / "val.parquet", val)
    save_listings(settings.splits_dir / "test.parquet", test)
    save_eval_items(settings.golden_dir / "used_goods.parquet", to_eval_items(test))
    save_json(
        settings.reports_dir / "split_report.json",
        {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "boundary_near_dupes": n_dupe,
            "train_strata": stratify_report(train),
            "test_strata": stratify_report(test),
        },
    )
    typer.echo(f"train={len(train)} val={len(val)} test={len(test)}")


@app.command("token-budget")
def token_budget():
    """Write token-length histograms and CUTOFF table under reports/token_length/."""
    settings = _bootstrap()
    from priceengine.training.token_budget import analyze_token_budget

    result = analyze_token_budget(settings)
    typer.echo(json.dumps(result, indent=2))


@app.command("prep-dataset")
def prep_dataset(
    style: str = typer.Option("ours", help="Prompt style: ours | ed"),
    cutoff: int = typer.Option(110, help="Summary token CUTOFF"),
    push_repo: str = typer.Option("", help="Optional private HF dataset repo id"),
):
    """Build prompt/completion HF dataset from splits."""
    settings = _bootstrap()
    from priceengine.training.dataset_prep import build_sft_dataset

    counts = build_sft_dataset(
        settings, style=style, cutoff=cutoff, push_repo=push_repo or None
    )
    typer.echo(counts)


@app.command("eval-baselines")
def eval_baselines(
    golden: Path = typer.Option(
        Path("data/golden/items_lite.parquet"), help="Golden-set parquet"
    ),
    train: Path = typer.Option(
        Path("data/splits/train.parquet"), help="Train parquet for category medians"
    ),
    limit: int = typer.Option(0, help="Optional cap on golden items (0 = all)"),
):
    """Grade dumb baselines on the golden set (no GPU). Writes reports/leaderboard-baselines.md."""
    settings = _bootstrap()
    from priceengine.corpus.io import load_eval_items, load_listings
    from priceengine.corpus.splits import to_eval_items
    from priceengine.eval.harness import run_pricer, write_leaderboard
    from priceengine.eval.pricers import CategoryMedianPricer, ConstantPricer

    items = load_eval_items(golden)
    if limit:
        items = items[:limit]
    train_items = to_eval_items(load_listings(train))
    global_median = sorted(i.price for i in train_items)[len(train_items) // 2]

    contestants = [
        CategoryMedianPricer(train_items),
        ConstantPricer(global_median, name="global-median"),
    ]
    results = {c.name: run_pricer(c, items) for c in contestants}
    path = write_leaderboard(
        battleground="items_lite_baselines",
        results=results,
        path=settings.reports_dir / "leaderboard-baselines.md",
    )
    typer.echo(f"Wrote {path}")


@app.command("eval-ed")
def eval_ed(
    golden: Path = typer.Option(
        Path("data/golden/items_lite.parquet"), help="Golden-set parquet"
    ),
    limit: int = typer.Option(100, help="How many test items (Modal cost scales with this)"),
    modal_app: str = typer.Option("pricer-service", help="Modal app name"),
):
    """Grade Ed's Modal specialist (R0) + baselines on items_lite golden set."""
    settings = _bootstrap()
    from priceengine.corpus.io import load_eval_items, load_listings
    from priceengine.corpus.splits import to_eval_items
    from priceengine.eval.ed_modal import ModalEdPricer
    from priceengine.eval.harness import run_pricer, write_leaderboard
    from priceengine.eval.pricers import CategoryMedianPricer, ConstantPricer

    items = load_eval_items(golden)
    if limit:
        items = items[:limit]
    train_items = to_eval_items(load_listings(settings.splits_dir / "train.parquet"))
    global_median = sorted(i.price for i in train_items)[len(train_items) // 2]

    ed = ModalEdPricer(modal_app=modal_app)
    contestants = [
        ed,
        CategoryMedianPricer(train_items),
        ConstantPricer(global_median, name="global-median"),
    ]
    results = {c.name: run_pricer(c, items) for c in contestants}
    path = write_leaderboard(
        battleground="items_lite",
        results=results,
        path=settings.reports_dir / "leaderboard-items_lite.md",
    )
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
