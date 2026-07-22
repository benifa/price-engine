"""Unit tests for cleaning, conditions, splits, and metrics (CPU only)."""

from datetime import date, timedelta

from priceengine.corpus.cleaning import (
    clean_listings,
    is_clean,
    listing_from_apify_item,
    round_price,
)
from priceengine.corpus.conditions import normalize_condition
from priceengine.corpus.splits import near_dupe_key, remove_boundary_near_dupes, time_split
from priceengine.eval.metrics import is_hit, paired_compare, summarize
from priceengine.models import Prediction, SoldListing


def _listing(
    item_id: str,
    price: float,
    sold: date,
    title: str = "Sony WH-1000XM5 Headphones",
    condition: str = "used-good",
) -> SoldListing:
    return SoldListing(
        item_id=item_id,
        title=title,
        description="Noise cancelling wireless headphones in great shape with case.",
        condition=condition,  # type: ignore[arg-type]
        category="Electronics",
        sold_price=price,
        sold_date=sold,
        listing_format="bin",
    )


def test_normalize_condition_ids_and_text():
    assert normalize_condition(1000) == "new"
    assert normalize_condition(3000) == "used-good"
    assert normalize_condition(6000) == "for-parts"
    assert normalize_condition("Open Box") == "open-box"
    assert normalize_condition("Seller refurbished") == "refurb"


def test_is_clean_rejects_out_of_range_and_thin():
    ok, _ = is_clean(_listing("1", 50, date(2026, 7, 1)))
    assert ok
    bad = _listing("2", 1500, date(2026, 7, 1))
    assert is_clean(bad)[0] is False
    thin = SoldListing(
        item_id="3",
        title="TV",
        description="x",
        condition="used-good",
        category="Electronics",
        sold_price=100,
        sold_date=date(2026, 7, 1),
    )
    assert is_clean(thin)[0] is False


def test_round_price_clamps():
    assert round_price(49.6) == 50
    assert round_price(0.2) == 1


def test_listing_from_apify_item():
    item = {
        "itemId": "abc",
        "title": "Dyson V15 Detect",
        "description": "Cordless vacuum, lightly used, all attachments included.",
        "soldPrice": "$329.00",
        "soldDate": "2026-07-10T12:00:00Z",
        "condition": "Used - Good",
        "category": "Appliances",
        "url": "https://ebay.com/itm/abc",
    }
    listing = listing_from_apify_item(item)
    assert listing is not None
    assert listing.sold_price == 329.0
    assert listing.condition == "used-good"


def test_time_split_and_near_dupes():
    as_of = date(2026, 7, 21)
    listings = [
        _listing("old", 100, as_of - timedelta(days=30)),
        _listing("val", 110, as_of - timedelta(days=10)),
        _listing("test", 120, as_of - timedelta(days=2)),
        _listing("dupe-test", 130, as_of - timedelta(days=1), title="Sony WH-1000XM5 Headphones"),
    ]
    # Make dupe-test share near-dupe key with old
    listings[3] = listings[3].model_copy(
        update={"title": listings[0].title, "condition": listings[0].condition}
    )
    train, val, test = time_split(listings, as_of=as_of)
    assert len(train) == 1 and len(val) == 1 and len(test) == 2
    train, val, test, dropped = remove_boundary_near_dupes(train, val, test)
    assert dropped >= 1
    assert all(near_dupe_key(t) != near_dupe_key(train[0]) for t in test)


def test_clean_listings_dedupes():
    day = date(2026, 7, 1)
    raw = [_listing("same", 100, day), _listing("same", 105, day)]
    kept, drops = clean_listings(raw)
    assert len(kept) == 1
    assert drops["duplicate_id"] == 1


def test_metrics_hit_and_summarize():
    preds = [
        Prediction(item_id="a", guess=100, truth=100, error=0),
        Prediction(item_id="b", guess=50, truth=100, error=50),
    ]
    assert is_hit(0, 100)
    assert not is_hit(50, 100)
    m = summarize("toy", "test", preds, bootstrap=False)
    assert m.n == 2
    assert m.mae == 25.0
    assert m.hit_rate == 0.5


def test_paired_compare_victory_flag():
    baseline = [
        Prediction(item_id="1", guess=200, truth=100, error=100),
        Prediction(item_id="2", guess=200, truth=100, error=100),
        Prediction(item_id="3", guess=200, truth=100, error=100),
        Prediction(item_id="4", guess=200, truth=100, error=100),
    ]
    ours = [
        Prediction(item_id="1", guess=110, truth=100, error=10),
        Prediction(item_id="2", guess=110, truth=100, error=10),
        Prediction(item_id="3", guess=110, truth=100, error=10),
        Prediction(item_id="4", guess=110, truth=100, error=10),
    ]
    cmp_ = paired_compare("ours", "ed", ours, baseline, battleground="used")
    assert cmp_.delta_mae > 0
    assert cmp_.relative_improvement >= 0.25
    assert cmp_.victory is True
