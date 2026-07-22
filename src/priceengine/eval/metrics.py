"""Metrics and paired bootstrap comparison."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from priceengine.config import Settings, get_settings
from priceengine.models import ComparisonResult, Prediction, RunMetrics


def _ape(guess: float, truth: float) -> float:
    if truth <= 0:
        return 0.0 if guess == 0 else 1.0
    return abs(guess - truth) / truth


def is_hit(error: float, truth: float, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return error < settings.hit_abs_dollars or (
        truth > 0 and error / truth < settings.hit_rel_fraction
    )


def summarize(
    name: str,
    battleground: str,
    preds: Sequence[Prediction],
    *,
    settings: Settings | None = None,
    bootstrap: bool = True,
) -> RunMetrics:
    settings = settings or get_settings()
    if not preds:
        return RunMetrics(
            name=name,
            battleground=battleground,
            n=0,
            mae=0.0,
            median_ape=0.0,
            hit_rate=0.0,
            rmsle=0.0,
        )
    errors = np.array([p.error for p in preds], dtype=float)
    apes = np.array([_ape(p.guess, p.truth) for p in preds], dtype=float)
    hits = np.array([is_hit(p.error, p.truth, settings) for p in preds], dtype=float)
    truths = np.array([p.truth for p in preds], dtype=float)
    guesses = np.array([max(0.0, p.guess) for p in preds], dtype=float)
    rmsle = float(np.sqrt(np.mean((np.log1p(guesses) - np.log1p(truths)) ** 2)))

    ci_low = ci_high = None
    if bootstrap and len(preds) >= 10:
        ci_low, ci_high = bootstrap_mae_ci(errors, n_samples=settings.bootstrap_samples)

    return RunMetrics(
        name=name,
        battleground=battleground,
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
    rng = np.random.default_rng(seed)
    n = len(errors)
    means = np.empty(n_samples)
    for i in range(n_samples):
        sample = errors[rng.integers(0, n, n)]
        means[i] = sample.mean()
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def paired_compare(
    ours_name: str,
    baseline_name: str,
    ours: Sequence[Prediction],
    baseline: Sequence[Prediction],
    *,
    battleground: str,
    settings: Settings | None = None,
) -> ComparisonResult:
    """Paired bootstrap on (baseline_error - ours_error). Positive delta => we win."""
    settings = settings or get_settings()
    by_id_ours = {p.item_id: p for p in ours}
    by_id_base = {p.item_id: p for p in baseline}
    ids = sorted(set(by_id_ours) & set(by_id_base))
    if not ids:
        raise ValueError("No overlapping item ids for paired comparison")

    deltas = np.array(
        [by_id_base[i].error - by_id_ours[i].error for i in ids], dtype=float
    )
    ours_mae = float(np.mean([by_id_ours[i].error for i in ids]))
    base_mae = float(np.mean([by_id_base[i].error for i in ids]))
    delta_mae = base_mae - ours_mae
    relative = delta_mae / base_mae if base_mae > 0 else 0.0

    rng = np.random.default_rng(42)
    n = len(deltas)
    boot = np.empty(settings.bootstrap_samples)
    for i in range(settings.bootstrap_samples):
        boot[i] = deltas[rng.integers(0, n, n)].mean()
    ci_low = float(np.quantile(boot, 0.025))
    ci_high = float(np.quantile(boot, 0.975))
    victory = (
        relative >= settings.victory_relative_mae and ci_low > 0
    )

    return ComparisonResult(
        ours=ours_name,
        baseline=baseline_name,
        battleground=battleground,
        delta_mae=delta_mae,
        relative_improvement=relative,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
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
        f"**{c.ours}** vs **{c.baseline}** on `{c.battleground}` (n={c.n}): "
        f"ΔMAE=${c.delta_mae:,.2f} ({c.relative_improvement:.1%} relative), "
        f"95% CI [${c.ci_low:,.2f}, ${c.ci_high:,.2f}] — {flag}"
    )
