from priceengine.config import PRICE_PREFIX, PRICE_QUESTION
from priceengine.models import ProductListing
from priceengine.train.local_sft import prompt_completion_rows_from_products


class _FakeTokenizer:
    """Cheap stand-in: 1 char ≈ 1 token for truncation tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text)))

    def decode(self, tokens: list[int]) -> str:
        return "x" * len(tokens)


def test_prompt_completion_rows_from_products_shape():
    product = ProductListing(
        item_id="1",
        title="Schlage Knob",
        description=(
            "Title: Schlage\nCategory: Hardware\nBrand: Schlage\n"
            "Description: A bronze knob."
        ),
        category="Hardware",
        price=64.3,
    )
    rows, n_truncated = prompt_completion_rows_from_products(
        [product], _FakeTokenizer(), cutoff=500
    )
    assert n_truncated == 0
    assert len(rows) == 1
    assert rows[0]["completion"] == "64.00"
    assert rows[0]["prompt"].startswith(PRICE_QUESTION)
    assert rows[0]["prompt"].endswith(PRICE_PREFIX)
    assert "Title: Schlage" in rows[0]["prompt"]
