"""Tests for Plotly eval HTML helpers."""

from __future__ import annotations

import json
from pathlib import Path

from priceengine.eval.visualization import write_eval_html
from priceengine.models import Prediction


def _preds(*errors: float) -> list[Prediction]:
    out = []
    for i, err in enumerate(errors):
        truth = 100.0
        guess = truth + err if i % 2 == 0 else truth - err
        # Keep |guess - truth| == err
        guess = truth + err
        out.append(
            Prediction(
                item_id=f"amazon_lite:test:{i}",
                guess=guess,
                truth=truth,
                error=abs(guess - truth),
                truncated=False,
                category="Electronics",
                condition=None,
            )
        )
    return out


def test_write_eval_html_versioned(tmp_path: Path, monkeypatch):
    from priceengine import config as cfg

    reports = tmp_path / "reports"
    reports.mkdir()
    settings = cfg.get_settings()
    monkeypatch.setattr(settings, "reports_dir", reports)

    leaderboard = tmp_path / "leaderboard.json"
    payload = {
        "list_price_qlora": [p.model_dump() for p in _preds(10.0, 50.0, 5.0)],
        "ed-donner/price-2025-11-28 (Modal)": [
            p.model_dump() for p in _preds(8.0, 12.0, 4.0)
        ],
    }
    leaderboard.write_text(json.dumps(payload))

    golden = Path("data/golden/amazon.parquet")
    if not golden.exists():
        # Skip title join; write_eval_html still needs load_eval_items
        import pytest

        pytest.skip("golden parquet not present")

    out = write_eval_html(
        leaderboard,
        golden=golden,
        out=reports / "eval_report.html",
        eval_set="amazon",
        settings=settings,
        open_browser=False,
        version="v0.1.0",
        worst_n=2,
    )
    assert out.name == "eval_report-v0.1.0.html"
    html = out.read_text()
    assert "Worst 2 misses" in html
    assert "Overlay" in html or "overlay" in html.lower()
    assert (reports / "eval_report.html").exists()
