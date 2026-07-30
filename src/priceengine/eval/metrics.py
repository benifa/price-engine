"""Eval metrics and paired-bootstrap victory test (docs/EVAL.md).

Plain language
--------------
* MAE — average dollar miss
* Median APE — typical percent miss
* Hit rate — “close enough” (under $40 or under 20% of the true price)
* RMSLE — error on a log scale (fairer across cheap vs expensive items)

Victory vs published baseline: at least 25% better MAE **and** the paired
bootstrap CI still says the gain is positive (see ``paired_compare``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from priceengine.config import Settings, get_settings
from priceengine.models import ComparisonResult, Prediction, RunMetrics


def _absolute_percentage_error(estimate: float, actual: float) -> float:
    if actual <= 0:
        return 0.0 if estimate == 0 else 1.0
    return abs(estimate - actual) / actual


def is_hit(error: float, actual: float, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return error < settings.hit_abs_dollars or (
        actual > 0 and error / actual < settings.hit_rel_fraction
    )


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_samples: int,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_samples)
    for i in range(n_samples):
        means[i] = values[rng.integers(0, n, n)].mean()
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def summarize(
    name: str,
    eval_set: str,
    preds: Sequence[Prediction],
    *,
    settings: Settings | None = None,
    bootstrap: bool = True,
) -> RunMetrics:
    """Roll per-item predictions into one leaderboard row."""
    settings = settings or get_settings()
    if not preds:
        return RunMetrics(
            name=name,
            eval_set=eval_set,
            n=0,
            mae=0.0,
            median_ape=0.0,
            hit_rate=0.0,
            rmsle=0.0,
        )

    errors = np.array([p.error for p in preds], dtype=float)
    apes = np.array(
        [_absolute_percentage_error(p.estimate, p.actual) for p in preds], dtype=float
    )
    hits = np.array(
        [is_hit(p.error, p.actual, settings) for p in preds], dtype=float
    )
    actuals = np.array([p.actual for p in preds], dtype=float)
    estimates = np.array([max(0.0, p.estimate) for p in preds], dtype=float)
    rmsle = float(np.sqrt(np.mean((np.log1p(estimates) - np.log1p(actuals)) ** 2)))

    ci_low = ci_high = None
    if bootstrap and len(preds) >= 10:
        ci_low, ci_high = _bootstrap_mean_ci(
            errors, n_samples=settings.bootstrap_samples
        )

    return RunMetrics(
        name=name,
        eval_set=eval_set,
        n=len(preds),
        mae=float(errors.mean()),
        median_ape=float(np.median(apes)),
        hit_rate=float(hits.mean()),
        rmsle=rmsle,
        mae_ci_low=ci_low,
        mae_ci_high=ci_high,
    )


def bootstrap_mae_ci(
    errors: np.ndarray, *, n_samples: int = 10_000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """CI on mean absolute error (used by tests)."""
    return _bootstrap_mean_ci(errors, n_samples=n_samples, alpha=alpha, seed=seed)


def paired_compare(
    challenger_name: str,
    baseline_name: str,
    challenger: Sequence[Prediction],
    baseline: Sequence[Prediction],
    *,
    eval_set: str,
    settings: Settings | None = None,
) -> ComparisonResult:
    """Paired bootstrap on (baseline_error − challenger_error). Positive = win."""
    settings = settings or get_settings()
    by_challenger = {p.item_id: p for p in challenger}
    by_baseline = {p.item_id: p for p in baseline}
    shared_ids = sorted(set(by_challenger) & set(by_baseline))
    if not shared_ids:
        raise ValueError("No overlapping item ids for paired comparison")

    deltas = np.array(
        [
            by_baseline[item_id].error - by_challenger[item_id].error
            for item_id in shared_ids
        ],
        dtype=float,
    )
    challenger_mae = float(
        np.mean([by_challenger[item_id].error for item_id in shared_ids])
    )
    baseline_mae = float(
        np.mean([by_baseline[item_id].error for item_id in shared_ids])
    )
    delta_mae = baseline_mae - challenger_mae
    relative = delta_mae / baseline_mae if baseline_mae > 0 else 0.0

    ci_low, ci_high = _bootstrap_mean_ci(
        deltas, n_samples=settings.bootstrap_samples
    )
    victory = relative >= settings.victory_relative_mae and ci_low > 0

    return ComparisonResult(
        challenger=challenger_name,
        baseline=baseline_name,
        eval_set=eval_set,
        delta_mae=delta_mae,
        relative_improvement=relative,
        ci_low=ci_low,
        ci_high=ci_high,
        n=len(shared_ids),
        victory=victory,
    )


def format_metrics_row(m: RunMetrics) -> str:
    ci = ""
    if m.mae_ci_low is not None and m.mae_ci_high is not None:
        ci = f"  (95% CI ${m.mae_ci_low:,.2f}–${m.mae_ci_high:,.2f})"
    return (
        f"| {m.name} | {m.n} | ${m.mae:,.2f}{ci} | {m.median_ape:.1%} | "
        f"{m.hit_rate:.1%} | {m.rmsle:.3f} |"
    )


def format_comparison(c: ComparisonResult) -> str:
    flag = "VICTORY" if c.victory else "not yet"
    return (
        f"**{c.challenger}** vs **{c.baseline}** on `{c.eval_set}` (n={c.n}): "
        f"ΔMAE=${c.delta_mae:,.2f} ({c.relative_improvement:.1%} relative), "
        f"95% CI [${c.ci_low:,.2f}, ${c.ci_high:,.2f}] — {flag}"
    )
