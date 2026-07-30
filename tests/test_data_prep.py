from priceengine.data import clamp_usd_price, hub_row_to_product


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
    assert listing.price == 64.0  # rounded
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


def test_product_listing_compat_list_price_field():
    from priceengine.models import ProductListing

    listing = ProductListing.model_validate(
        {
            "item_id": "1",
            "title": "t",
            "description": "d",
            "category": "Other",
            "list_price": 10.0,
        }
    )
    assert listing.price == 10.0


def test_eval_item_compat_id_field():
    from priceengine.models import EvalItem

    item = EvalItem.model_validate(
        {
            "id": "abc",
            "title": "t",
            "description": "d",
            "price": 12.0,
        }
    )
    assert item.item_id == "abc"
