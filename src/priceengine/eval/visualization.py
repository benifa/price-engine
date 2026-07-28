"""Course-style Plotly eval report (truth vs guess + cumulative MAE).

Matches the week-7 ``Tester`` charts: green/orange/red by hit quality, ``y = x``
reference line, and a running-average error band. Writes a self-contained HTML
file you can open in a browser.
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
from priceengine.data_prep import load_eval_items
from priceengine.eval.metrics import is_hit, summarize
from priceengine.models import Prediction

logger = logging.getLogger(__name__)


def _hit_color(error: float, truth: float, settings: Settings) -> str:
    """Green = hit, orange = near miss, red = miss (same bands as the course)."""
    if is_hit(error, truth, settings):
        return "green"
    if error < 80 or (truth > 0 and error / truth < 0.4):
        return "orange"
    return "red"


def _load_predictions(path: Path) -> dict[str, list[Prediction]]:
    raw = json.loads(path.read_text())
    return {
        name: [Prediction.model_validate(row) for row in rows]
        for name, rows in raw.items()
    }


def _titles_by_id(golden: Path) -> dict[str, str]:
    return {item.id: item.title for item in load_eval_items(golden)}


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
                "model": name,
                "item_id": pred.item_id,
                "title": short,
                "truth": pred.truth,
                "guess": pred.guess,
                "error": pred.error,
                "category": pred.category or "",
                "color": _hit_color(pred.error, pred.truth, settings),
            }
        )
    frame = pd.DataFrame(rows)
    frame["hover"] = [
        f"{t}<br>Guess=${g:,.2f} Actual=${y:,.2f} Error=${e:,.2f}"
        for t, g, y, e in zip(
            frame["title"], frame["guess"], frame["truth"], frame["error"], strict=True
        )
    ]
    return frame


def _scatter_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    max_val = float(max(frame["truth"].max(), frame["guess"].max(), 1.0))
    fig = px.scatter(
        frame,
        x="truth",
        y="guess",
        color="color",
        color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
        title=title,
        labels={"truth": "Actual price ($)", "guess": "Predicted price ($)"},
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
        title=f"{model_name} — running MAE ${final_mean:,.2f} ± ${final_ci:,.2f}",
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
    names = []
    maes = []
    for name, preds in results.items():
        metrics = summarize(name, eval_set, preds, settings=settings, bootstrap=False)
        names.append(name)
        maes.append(metrics.mae)
    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=maes,
                text=[f"${m:,.2f}" for m in maes],
                textposition="outside",
                marker_color="steelblue",
            )
        ]
    )
    fig.update_layout(
        title=f"MAE by model — `{eval_set}`",
        xaxis_title="Model",
        yaxis_title="MAE ($)",
        width=900,
        height=400,
        template="plotly_white",
        margin=dict(t=60, b=120),
    )
    return fig


def write_eval_html(
    leaderboard_json: Path,
    *,
    golden: Path,
    out: Path,
    eval_set: str | None = None,
    settings: Settings | None = None,
    open_browser: bool = False,
) -> Path:
    """Build a multi-model HTML report from ``leaderboard.json``."""
    settings = settings or get_settings()
    results = _load_predictions(leaderboard_json)
    if not results:
        raise ValueError(f"No predictions in {leaderboard_json}")

    eval_set = eval_set or leaderboard_json.stem.replace("leaderboard-", "") or "amazon"
    titles = _titles_by_id(golden)

    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Price Engine — eval report</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:960px}"
        "h1,h2{font-weight:600} .meta{color:#555;margin-bottom:2rem}</style>",
        "</head><body>",
        "<h1>Price Engine — eval report</h1>",
        f"<p class='meta'>Source: <code>{leaderboard_json}</code> · "
        f"eval set <code>{eval_set}</code> · "
        "green = hit (&lt;$40 or &lt;20%), orange = near, red = miss</p>",
    ]

    mae_fig = _mae_bar_figure(results, settings, eval_set)
    parts.append(mae_fig.to_html(full_html=False, include_plotlyjs="cdn"))

    for name, preds in results.items():
        metrics = summarize(name, eval_set, preds, settings=settings, bootstrap=False)
        frame = _frame_for_model(name, preds, titles, settings)
        heading = (
            f"{name} — MAE ${metrics.mae:,.2f} · hit {metrics.hit_rate:.0%} · "
            f"n={metrics.n}"
        )
        parts.append(f"<h2>{heading}</h2>")
        scatter = _scatter_figure(
            frame,
            title=(
                f"{name}<br><b>MAE:</b> ${metrics.mae:,.2f} "
                f"<b>Hit:</b> {metrics.hit_rate:.0%} <b>RMSLE:</b> {metrics.rmsle:.3f}"
            ),
        )
        trend = _error_trend_figure([p.error for p in preds], name)
        # First model includes plotly.js from MAE chart; later figs omit the bundle.
        parts.append(scatter.to_html(full_html=False, include_plotlyjs=False))
        parts.append(trend.to_html(full_html=False, include_plotlyjs=False))

    parts.append("</body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    logger.info("Wrote eval HTML → %s", out)
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out
