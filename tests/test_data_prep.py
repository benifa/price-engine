from priceengine.data_prep import clamp_usd_price, hub_row_to_product


def test_hub_row_to_product():
    row = {
        "title": "Schlage Knob",
        "summary": (
            "Title: Schlage\nCategory: Hardware\nBrand: Schlage\n"
            "Description: A bronze knob."
        ),
        "category": "Tools_and_Home_Improvement",
        "price": 64.3,
        "id": 42,
    }
    listing = hub_row_to_product(row, split="train", index=0)
    assert listing is not None
    assert listing.list_price == 64.0  # rounded
    assert listing.condition == "new"
    assert listing.item_id == "42"


def test_hub_row_rejects_out_of_range():
    row = {
        "title": "x",
        "summary": "long enough description here",
        "price": 2000,
    }
    assert hub_row_to_product(row, split="t", index=0) is None


def test_clamp_usd_price():
    assert clamp_usd_price(49.6) == 50
    assert clamp_usd_price(0.2) == 1
