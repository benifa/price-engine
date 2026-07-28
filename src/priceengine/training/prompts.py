"""List-price prompt and completion text (training + eval).

Format (``amazon_list``)
------------------------
::

    What does this cost to the nearest dollar?

    <product title + description>

    Price is $

The model then emits ``NNN.00`` (integer dollars in [$1, $999]). That matches the
published baseline and Hub rows in ``ed-donner/items_prompts_full``.

``truncate_text`` is applied to the product body *before* wrapping so the
question + prefix always fit; CUTOFF defaults to ``SUMMARY_CUTOFF`` (110).
"""

from __future__ import annotations

from priceengine.config import AMAZON_LIST_QUESTION, PRICE_PREFIX
from priceengine.models import EvalItem, ProductListing


def truncate_text(text: str, tokenizer, max_tokens: int) -> tuple[str, bool]:
    """Truncate with the model tokenizer. Returns ``(text, was_truncated)``."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text, False
    truncated = tokenizer.decode(tokens[:max_tokens]).rstrip()
    return truncated, True


def price_completion(price: float) -> str:
    """Dollar amount the model should emit after ``Price is $``."""
    return f"{round(price)}.00"


def list_price_prompt(text: str) -> str:
    """Full prompt up through ``Price is $`` (completion starts immediately after)."""
    return f"{AMAZON_LIST_QUESTION}\n\n{text}\n\n{PRICE_PREFIX}"


def prompt_for_eval_item(item: EvalItem, *, text: str | None = None) -> str:
    """Build an eval prompt; pass ``text`` when the caller already truncated."""
    body = text if text is not None else item.text_for_pricing()
    return list_price_prompt(body)


def training_example(product: ProductListing, *, text: str) -> dict[str, str]:
    """One SFT row: ``{"prompt": ..., "completion": ...}`` for TRL / Hub datasets."""
    return {
        "prompt": list_price_prompt(text),
        "completion": price_completion(product.list_price),
    }
