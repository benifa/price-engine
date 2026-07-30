"""Frontier chat models (OpenAI) as dollar pricers for the eval leaderboard.

Uses the same list-price prompt as fine-tuned adapters. Requires ``OPENAI_API_KEY``.
Optional dep: ``openai`` (``uv sync --extra frontier``).
"""

from __future__ import annotations

import logging
import os
import re

from priceengine.eval.pricers import extract_price
from priceengine.models import EvalItem
from priceengine.prompts import prompt_for_eval_item, truncate_text

logger = logging.getLogger(__name__)

DEFAULT_FRONTIER_MODEL = "gpt-5"


def frontier_display_name(model: str) -> str:
    return f"Frontier · {model}"


class FrontierChatPricer:
    """Call an OpenAI chat model; parse the first dollar amount from the reply."""

    def __init__(
        self,
        model: str = DEFAULT_FRONTIER_MODEL,
        *,
        name: str | None = None,
        cutoff_tokens: int | None = None,
        tokenizer_name: str | None = None,
        max_completion_tokens: int = 32,
    ):
        from transformers import AutoTokenizer

        from priceengine.config import BASE_MODEL, MAX_DESCRIPTION_TOKENS

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for --frontier "
                "(add it to .env or the environment)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install the frontier extra: uv sync --extra frontier"
            ) from exc

        self.model = model
        self.name = name or frontier_display_name(model)
        self.max_completion_tokens = max_completion_tokens
        self._client = OpenAI(api_key=api_key)
        self._cutoff = (
            MAX_DESCRIPTION_TOKENS if cutoff_tokens is None else cutoff_tokens
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or BASE_MODEL
        )
        logger.info("Frontier pricer ready: %s", self.name)

    def _prompt(self, item: EvalItem) -> str:
        body, _ = truncate_text(
            item.text_for_pricing(), self._tokenizer, self._cutoff
        )
        return prompt_for_eval_item(item, text=body)

    def price(self, item: EvalItem) -> float:
        prompt = self._prompt(item)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You estimate Amazon list prices in US dollars. "
                        "Reply with only a number like 129.00 — no words."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=self.max_completion_tokens,
            temperature=0,
        )
        text = (response.choices[0].message.content or "").strip()
        # Models sometimes wrap as "$129" or "Price is $129.00"
        if not re.search(r"\d", text):
            logger.warning("Frontier empty/unparseable reply for %s: %r", item.item_id, text)
            return 0.0
        return extract_price(text)
