"""Unit tests for eval metrics (CPU only)."""

from priceengine.eval.metrics import is_hit, paired_compare, summarize
from priceengine.models import Prediction


def test_hit_and_summarize():
    preds = [
        Prediction(item_id="a", guess=100, truth=100, error=0),
        Prediction(item_id="b", guess=50, truth=100, error=50),
    ]
    assert is_hit(0, 100)
    assert not is_hit(50, 100)
    m = summarize("toy", "amazon", preds, bootstrap=False)
    assert m.n == 2
    assert m.mae == 25.0
    assert m.eval_set == "amazon"


def test_paired_compare_victory_flag():
    baseline = [
        Prediction(item_id="1", guess=200, truth=100, error=100),
        Prediction(item_id="2", guess=200, truth=100, error=100),
        Prediction(item_id="3", guess=200, truth=100, error=100),
        Prediction(item_id="4", guess=200, truth=100, error=100),
    ]
    challenger = [
        Prediction(item_id="1", guess=110, truth=100, error=10),
        Prediction(item_id="2", guess=110, truth=100, error=10),
        Prediction(item_id="3", guess=110, truth=100, error=10),
        Prediction(item_id="4", guess=110, truth=100, error=10),
    ]
    cmp_ = paired_compare(
        "challenger", "baseline", challenger, baseline, eval_set="amazon"
    )
    assert cmp_.victory
    assert cmp_.relative_improvement > 0.25
    assert cmp_.eval_set == "amazon"
