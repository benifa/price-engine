"""Plotly eval report (estimate vs actual + cumulative MAE).

Top of the page is a ranked “who wins” table with clear roles
(our model / frontier / published specialist / naive floors), then charts.
"""

from __future__ import annotations

import json
import logging
import math
import webbrowser
from itertools import accumulate
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from priceengine.config import Settings, get_settings
from priceengine.data import load_eval_items
from priceengine.eval.baseline_pricer import BASELINE_NAME
from priceengine.eval.metrics import is_hit, summarize
from priceengine.eval.roles import (
    ROLE_FRONTIER,
    ROLE_NAIVE,
    ROLE_OURS,
    ROLE_SPECIALIST,
    blurb_for,
    is_naive_floor,
    role_color,
    role_for,
    short_label,
)
from priceengine.models import Prediction

logger = logging.getLogger(__name__)

_OVERLAY_PRIORITY = (
    "list_price_qlora",
    BASELINE_NAME,
)


def _hit_color(error: float, actual: float, settings: Settings) -> str:
    """Green = hit, orange = near miss, red = miss."""
    if is_hit(error, actual, settings):
        return "green"
    if error < 80 or (actual > 0 and error / actual < 0.4):
        return "orange"
    return "red"


def _load_predictions(path: Path) -> dict[str, list[Prediction]]:
    raw = json.loads(path.read_text())
    return {
        name: [Prediction.model_validate(row) for row in rows]
        for name, rows in raw.items()
    }


def _titles_by_id(golden: Path) -> dict[str, str]:
    return {item.item_id: item.title for item in load_eval_items(golden)}


def _frame_for_model(
    name: str,
    preds: list[Prediction],
    titles: dict[str, str],
    settings: Settings,
) -> pd.DataFrame:
    rows = []
    for pred in preds:
        title = titles.get(pred.item_id, pred.item_id)
        short = title if len(title) <= 40 else title[:40] + "..."
        rows.append(
            {
                "model": short_label(name),
                "item_id": pred.item_id,
                "title": short,
                "full_title": title,
                "actual": pred.actual,
                "estimate": pred.estimate,
                "error": pred.error,
                "category": pred.category or "",
                "color": _hit_color(pred.error, pred.actual, settings),
            }
        )
    frame = pd.DataFrame(rows)
    frame["hover"] = [
        f"{t}<br>Estimate=${g:,.2f} Actual=${y:,.2f} Error=${e:,.2f}"
        for t, g, y, e in zip(
            frame["title"], frame["estimate"], frame["actual"], frame["error"], strict=True
        )
    ]
    return frame


def _ranked_metrics(
    results: dict[str, list[Prediction]], settings: Settings, eval_set: str
) -> list[tuple[str, object]]:
    rows = [
        (name, summarize(name, eval_set, preds, settings=settings, bootstrap=False))
        for name, preds in results.items()
    ]
    rows.sort(key=lambda pair: pair[1].mae)
    return rows


def _presence_note(results: dict[str, list[Prediction]]) -> str:
    roles = {role_for(name) for name in results}
    missing: list[str] = []
    if ROLE_OURS not in roles:
        missing.append("our fine-tuned model (<code>--adapter-path</code>)")
    if ROLE_FRONTIER not in roles:
        missing.append("a frontier API model (<code>--frontier gpt-5</code>)")
    if ROLE_SPECIALIST not in roles:
        missing.append("the published specialist (default; skip with <code>--no-include-baseline</code>)")
    if not missing:
        return (
            "<p class='callout ok'>Comparing our model, frontier APIs, the published "
            "specialist, and naive floors on the same held-out products.</p>"
        )
    return (
        "<p class='callout warn'><strong>Incomplete board.</strong> "
        "Not scored yet: " + "; ".join(missing) + ". "
        "Naive floors alone are a sanity check — not the headline result.</p>"
    )


def _summary_table_html(
    results: dict[str, list[Prediction]], settings: Settings, eval_set: str
) -> str:
    ranked = _ranked_metrics(results, settings, eval_set)
    rows = [
        "<table class='board'>",
        "<thead><tr>"
        "<th>Rank</th><th>Role</th><th>Model</th><th>MAE ↓</th>"
        "<th>Hit rate</th><th>n</th><th>What this is</th>"
        "</tr></thead><tbody>",
    ]
    for rank, (name, metrics) in enumerate(ranked, start=1):
        role = role_for(name)
        color = role_color(role)
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td><span class='role' style='background:{color}'>{_escape(role)}</span></td>"
            f"<td><strong>{_escape(short_label(name))}</strong></td>"
            f"<td>${metrics.mae:,.2f}</td>"
            f"<td>{metrics.hit_rate:.0%}</td>"
            f"<td>{metrics.n}</td>"
            f"<td class='blurb'>{_escape(blurb_for(name))}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _scatter_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    max_val = float(max(frame["actual"].max(), frame["estimate"].max(), 1.0))
    fig = px.scatter(
        frame,
        x="actual",
        y="estimate",
        color="color",
        color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
        title=title,
        labels={"actual": "Actual price ($)", "estimate": "Predicted price ($)"},
        category_orders={"color": ["green", "orange", "red"]},
        width=900,
        height=700,
    )
    for trace in fig.data:
        mask = frame["color"] == trace.name
        trace.customdata = frame.loc[mask, ["hover"]].to_numpy()
        trace.hovertemplate = "%{customdata[0]}<extra></extra>"
        trace.marker.update(size=7)
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(width=2, dash="dash", color="deepskyblue"),
            name="y = x",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_xaxes(range=[0, max_val])
    fig.update_yaxes(range=[0, max_val])
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def _overlay_scatter(
    results: dict[str, list[Prediction]],
    titles: dict[str, str],
    settings: Settings,
) -> go.Figure | None:
    """Our model vs published baseline / frontier on one chart when present."""
    names = [n for n in _OVERLAY_PRIORITY if n in results]
    if len(names) < 2:
        names = [n for n in results if not is_naive_floor(n)][:2]
    if len(names) < 2:
        return None

    frames = []
    for name in names:
        frame = _frame_for_model(name, results[name], titles, settings)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    max_val = float(max(combined["actual"].max(), combined["estimate"].max(), 1.0))

    fig = px.scatter(
        combined,
        x="actual",
        y="estimate",
        color="model",
        title="Overlay — predicted vs actual (models, not naive floors)",
        labels={"actual": "Actual price ($)", "estimate": "Predicted price ($)"},
        hover_data=["title", "error"],
        width=900,
        height=700,
        opacity=0.75,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(width=2, dash="dash", color="deepskyblue"),
            name="y = x",
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(range=[0, max_val])
    fig.update_yaxes(range=[0, max_val])
    fig.update_layout(template="plotly_white")
    return fig


def _error_trend_figure(errors: list[float], model_name: str) -> go.Figure:
    n = len(errors)
    x = list(range(1, n + 1))
    running_sums = list(accumulate(errors))
    running_means = [s / i for s, i in zip(running_sums, x, strict=True)]
    running_squares = list(accumulate(e * e for e in errors))
    running_stds = [
        math.sqrt(max(0.0, (sq / i) - (mean**2))) if i > 1 else 0.0
        for i, sq, mean in zip(x, running_squares, running_means, strict=True)
    ]
    ci = [
        1.96 * (sd / math.sqrt(i)) if i > 1 else 0.0
        for i, sd in zip(x, running_stds, strict=True)
    ]
    upper = [m + c for m, c in zip(running_means, ci, strict=True)]
    lower = [m - c for m, c in zip(running_means, ci, strict=True)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(128,128,128,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=running_means,
            mode="lines",
            line=dict(width=3, color="firebrick"),
            name="Cumulative avg error",
            customdata=[[c] for c in ci],
            hovertemplate=(
                "n=%{x}<br>Avg error=$%{y:,.2f}<br>"
                "±95% CI=$%{customdata[0]:,.2f}<extra></extra>"
            ),
        )
    )
    final_mean = running_means[-1]
    final_ci = ci[-1]
    fig.update_layout(
        title=f"{short_label(model_name)} — running MAE ${final_mean:,.2f} ± ${final_ci:,.2f}",
        xaxis_title="Number of items",
        yaxis_title="Average absolute error ($)",
        width=900,
        height=360,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def _mae_bar_figure(
    results: dict[str, list[Prediction]], settings: Settings, eval_set: str
) -> go.Figure:
    ranked = _ranked_metrics(results, settings, eval_set)
    labels = [f"{short_label(name)}\n({role_for(name)})" for name, _ in ranked]
    maes = [metrics.mae for _, metrics in ranked]
    colors = [role_color(role_for(name)) for name, _ in ranked]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=maes,
                text=[f"${m:,.2f}" for m in maes],
                textposition="outside",
                marker_color=colors,
            )
        ]
    )
    fig.update_layout(
        title=f"Lower MAE is better — `{eval_set}`",
        xaxis_title="",
        yaxis_title="Mean absolute error ($)",
        width=900,
        height=420,
        template="plotly_white",
        margin=dict(t=60, b=140),
    )
    return fig


def _worst_misses_html(
    name: str,
    preds: list[Prediction],
    titles: dict[str, str],
    *,
    top_n: int = 20,
) -> str:
    ranked = sorted(preds, key=lambda p: p.error, reverse=True)[:top_n]
    rows = [
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;width:100%;font-size:14px'>",
        "<thead><tr>"
        "<th>#</th><th>Title</th><th>Actual</th><th>Estimate</th><th>Error</th>"
        "</tr></thead><tbody>",
    ]
    for i, pred in enumerate(ranked, start=1):
        title = titles.get(pred.item_id, pred.item_id)
        if len(title) > 60:
            title = title[:60] + "…"
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{_escape(title)}</td>"
            f"<td>${pred.actual:,.2f}</td>"
            f"<td>${pred.estimate:,.2f}</td>"
            f"<td>${pred.error:,.2f}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return (
        f"<h3>Worst {top_n} misses — {_escape(short_label(name))}</h3>\n"
        + "\n".join(rows)
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_eval_html(
    leaderboard_json: Path,
    *,
    golden: Path,
    out: Path,
    eval_set: str | None = None,
    settings: Settings | None = None,
    open_browser: bool = False,
    version: str | None = None,
    worst_n: int = 20,
) -> Path:
    """Build a multi-model HTML report from ``leaderboard.json``.

    When ``version`` is set (e.g. ``v0.1.0``), also copies the report to
    ``reports/eval_report-{version}.html`` beside ``out`` if ``out`` is the
    default stem, or uses ``out`` as-is when the caller already versioned it.
    """
    settings = settings or get_settings()
    results = _load_predictions(leaderboard_json)
    if not results:
        raise ValueError(f"No predictions in {leaderboard_json}")

    eval_set = eval_set or leaderboard_json.stem.replace("leaderboard-", "") or "list-price golden"
    titles = _titles_by_id(golden)

    if version and out.name == "eval_report.html":
        out = out.with_name(f"eval_report-{version}.html")

    version_note = f" · version <code>{version}</code>" if version else ""
    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Price Engine — who wins on list price?</title>",
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:980px;color:#0f172a}"
        "h1,h2,h3{font-weight:650}"
        ".meta{color:#64748b;margin:0.5rem 0 1.25rem}"
        ".callout{padding:0.85rem 1rem;border-radius:8px;margin:1rem 0 1.5rem;line-height:1.45}"
        ".callout.warn{background:#fff7ed;border:1px solid #fdba74}"
        ".callout.ok{background:#ecfdf5;border:1px solid #6ee7b7}"
        "table.board{border-collapse:collapse;width:100%;font-size:14px;margin:0.5rem 0 1.5rem}"
        "table.board th,table.board td{border-bottom:1px solid #e2e8f0;padding:0.55rem 0.45rem;text-align:left;vertical-align:top}"
        "table.board th{background:#f8fafc;font-size:12px;text-transform:uppercase;letter-spacing:0.03em;color:#475569}"
        ".role{display:inline-block;color:#fff;font-size:12px;font-weight:600;padding:0.15rem 0.45rem;border-radius:4px}"
        ".blurb{color:#64748b;font-size:13px;max-width:22rem}"
        "table{margin:1rem 0 2rem} th{background:#f4f4f4;text-align:left}"
        "</style>",
        "</head><body>",
        "<h1>Who wins on list price?</h1>",
        f"<p class='meta'>Held-out set: <code>{_escape(eval_set)}</code> · "
        f"source <code>{_escape(str(leaderboard_json))}</code>{version_note}<br>"
        "Hit = absolute error &lt;$40 <em>or</em> relative error &lt;20% "
        "(green / orange / red on charts below).</p>",
        _presence_note(results),
        "<h2>Ranked comparison</h2>",
        "<p class='meta'>Lower MAE is better. Roles: "
        f"<strong>{ROLE_OURS}</strong>, <strong>{ROLE_FRONTIER}</strong>, "
        f"<strong>{ROLE_SPECIALIST}</strong>, <strong>{ROLE_NAIVE}</strong> "
        "(category / train median guesses with no neural net).</p>",
        _summary_table_html(results, settings, eval_set),
    ]

    mae_fig = _mae_bar_figure(results, settings, eval_set)
    parts.append(mae_fig.to_html(full_html=False, include_plotlyjs="cdn"))

    overlay = _overlay_scatter(results, titles, settings)
    if overlay is not None:
        parts.append("<h2>Model overlay</h2>")
        parts.append(overlay.to_html(full_html=False, include_plotlyjs=False))

    parts.append("<h2>Per-model detail</h2>")
    for name, preds in results.items():
        metrics = summarize(name, eval_set, preds, settings=settings, bootstrap=False)
        frame = _frame_for_model(name, preds, titles, settings)
        label = short_label(name)
        role = role_for(name)
        heading = (
            f"{_escape(label)} <span class='role' style='background:{role_color(role)}'>"
            f"{_escape(role)}</span> — MAE ${metrics.mae:,.2f} · "
            f"hit {metrics.hit_rate:.0%} · n={metrics.n}"
        )
        parts.append(f"<h3>{heading}</h3>")
        if blurb_for(name):
            parts.append(f"<p class='meta'>{_escape(blurb_for(name))}</p>")
        scatter = _scatter_figure(
            frame,
            title=(
                f"{label}<br><b>MAE:</b> ${metrics.mae:,.2f} "
                f"<b>Hit:</b> {metrics.hit_rate:.0%} <b>RMSLE:</b> {metrics.rmsle:.3f}"
            ),
        )
        trend = _error_trend_figure([p.error for p in preds], name)
        parts.append(scatter.to_html(full_html=False, include_plotlyjs=False))
        parts.append(trend.to_html(full_html=False, include_plotlyjs=False))
        parts.append(_worst_misses_html(name, preds, titles, top_n=worst_n))

    parts.append("</body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    logger.info("Wrote eval HTML → %s", out)

    latest = settings.reports_dir / "eval_report.html"
    if out.resolve() != latest.resolve():
        latest.write_text(out.read_text())
        logger.info("Also wrote %s", latest)

    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out
