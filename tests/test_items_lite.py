from priceengine.corpus.items_lite import row_to_listing


def test_row_to_listing_items_lite():
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
    listing = row_to_listing(row, split="train", index=0)
    assert listing is not None
    assert listing.sold_price == 64.0  # rounded
    assert listing.condition == "new"
    assert listing.item_id == "42"


def test_row_rejects_out_of_range():
    row = {
        "title": "x",
        "summary": "long enough description here",
        "price": 2000,
    }
    assert row_to_listing(row, split="t", index=0) is None
