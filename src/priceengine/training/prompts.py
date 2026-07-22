"""Prompt construction for Ed's format and ours — shared by training and eval."""

from __future__ import annotations

from priceengine.config import ED_QUESTION, OURS_QUESTION, PRICE_PREFIX
from priceengine.models import EvalItem, SoldListing


def truncate_text(text: str, tokenizer, max_tokens: int) -> tuple[str, bool]:
    """Truncate with the model tokenizer. Returns (text, was_truncated)."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text, False
    truncated = tokenizer.decode(tokens[:max_tokens]).rstrip()
    return truncated, True


def ed_prompt(text: str) -> str:
    return f"{ED_QUESTION}\n\n{text}\n\n{PRICE_PREFIX}"


def ed_completion(price: float) -> str:
    return f"{round(price)}.00"


def ours_header(condition: str, category: str, sold_ym: str) -> str:
    return f"Condition: {condition} | Category: {category} | Sold: {sold_ym}"


def ours_prompt(text: str, *, condition: str, category: str, sold_ym: str) -> str:
    header = ours_header(condition, category, sold_ym)
    return f"{OURS_QUESTION}\n\n{header}\n\n{text}\n\n{PRICE_PREFIX}"


def ours_completion(price: float) -> str:
    return f"{round(price)}.00"


def prompt_for_eval_item(item: EvalItem, *, style: str, text: str | None = None) -> str:
    """Build the inference prompt (no completion) for an eval item."""
    body = text if text is not None else item.text_for_pricing()
    if style == "ed":
        return ed_prompt(body)
    if style == "ours":
        sold_ym = item.sold_date.strftime("%Y-%m") if item.sold_date else "unknown"
        return ours_prompt(
            body,
            condition=item.condition or "used-good",
            category=item.category or "Other",
            sold_ym=sold_ym,
        )
    raise ValueError(f"Unknown prompt style: {style}")


def training_example(listing: SoldListing, *, style: str, text: str) -> dict[str, str]:
    if style == "ed":
        return {"prompt": ed_prompt(text), "completion": ed_completion(listing.sold_price)}
    if style == "ours":
        return {
            "prompt": ours_prompt(
                text,
                condition=listing.condition,
                category=listing.category,
                sold_ym=listing.sold_date.strftime("%Y-%m"),
            ),
            "completion": ours_completion(listing.sold_price),
        }
    raise ValueError(f"Unknown prompt style: {style}")
