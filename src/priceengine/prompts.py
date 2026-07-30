"""List-price prompt text shared by train and eval.

Format::

    What does this cost to the nearest dollar?

    <title + description>

    Price is $

The model continues with ``NNN.00`` (whole dollars in $1–$999).
"""

from __future__ import annotations

from priceengine.config import PRICE_PREFIX, PRICE_QUESTION
from priceengine.models import EvalItem, ProductListing


def truncate_text(text: str, tokenizer, max_tokens: int) -> tuple[str, bool]:
    """Keep at most ``max_tokens`` tokens. Returns (text, was_truncated)."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text, False
    return tokenizer.decode(tokens[:max_tokens]).rstrip(), True


def price_completion(price: float) -> str:
    """Text the model should emit after ``Price is $``."""
    return f"{round(price)}.00"


def list_price_prompt(text: str) -> str:
    """Full prompt ending at ``Price is $`` (completion starts right after)."""
    return f"{PRICE_QUESTION}\n\n{text}\n\n{PRICE_PREFIX}"


def prompt_for_eval_item(item: EvalItem, *, text: str | None = None) -> str:
    """Build an eval prompt. Pass ``text`` if you already truncated the body."""
    body = text if text is not None else item.text_for_pricing()
    return list_price_prompt(body)


def training_example(product: ProductListing, *, text: str) -> dict[str, str]:
    """One SFT row: prompt + completion for TRL / Hub datasets."""
    return {
        "prompt": list_price_prompt(text),
        "completion": price_completion(product.price),
    }
