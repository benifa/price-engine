"""Published baseline via Modal ``pricer-service`` (ed-donner/price-2025-11-28).

We call Modal instead of loading 4-bit weights on macOS. Send a *truncated
description only* — the remote wrapper adds the question and ``Price is $``.
"""

from __future__ import annotations

import logging

from priceengine.models import EvalItem
from priceengine.prompts import truncate_text

logger = logging.getLogger(__name__)

BASELINE_NAME = "ed-donner/price-2025-11-28 (Modal)"


class PublishedBaselinePricer:
    """Remote published adapter — fair-eval reference model."""

    name = BASELINE_NAME

    def __init__(
        self,
        *,
        modal_app: str = "pricer-service",
        modal_class: str = "Pricer",
        cutoff_tokens: int | None = None,
        tokenizer_name: str | None = None,
    ):
        import modal
        from transformers import AutoTokenizer

        from priceengine.config import BASE_MODEL, MAX_DESCRIPTION_TOKENS

        remote_cls = modal.Cls.from_name(modal_app, modal_class)
        self._remote = remote_cls()
        self._cutoff_tokens = (
            MAX_DESCRIPTION_TOKENS if cutoff_tokens is None else cutoff_tokens
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or BASE_MODEL
        )
        logger.info("Connected to Modal %s/%s", modal_app, modal_class)

    def _description(self, item: EvalItem) -> str:
        text, _ = truncate_text(
            item.text_for_pricing(), self._tokenizer, self._cutoff_tokens
        )
        return text

    def price(self, item: EvalItem) -> float:
        return float(self._remote.price.remote(self._description(item)))
