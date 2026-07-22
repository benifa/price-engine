"""Evaluate Ed's published specialist via the existing Modal pricer service.

Avoids loading 4-bit weights locally (bitsandbytes is awkward on macOS).
Uses the same remote class Deal Hunter already deploys.
"""

from __future__ import annotations

import logging

from priceengine.config import ED_QUESTION, PRICE_PREFIX
from priceengine.models import EvalItem
from priceengine.training.prompts import truncate_text

logger = logging.getLogger(__name__)


class ModalEdPricer:
    """R0 contestant: ed-donner adapter served on Modal (`pricer-service` / `Pricer`)."""

    name = "ed-donner/price-2025-11-28 (Modal)"

    def __init__(
        self,
        *,
        modal_app: str = "pricer-service",
        modal_class: str = "Pricer",
        cutoff_tokens: int = 110,
        tokenizer_name: str = "meta-llama/Llama-3.2-3B",
    ):
        import modal
        from transformers import AutoTokenizer

        Pricer = modal.Cls.from_name(modal_app, modal_class)
        self.pricer = Pricer()
        self.cutoff_tokens = cutoff_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info("Connected to Modal %s/%s", modal_app, modal_class)

    def _ed_description(self, item: EvalItem) -> str:
        """Match Ed's serving path: truncated summary text only (no question wrapper).

        The Modal Pricer builds: QUESTION + description + PREFIX internally.
        """
        text, _ = truncate_text(item.text_for_pricing(), self.tokenizer, self.cutoff_tokens)
        return text

    def price(self, item: EvalItem) -> float:
        return float(self.pricer.price.remote(self._ed_description(item)))


# Re-export constants so callers can build matching local prompts if needed
ED_LOCAL_QUESTION = ED_QUESTION
ED_LOCAL_PREFIX = PRICE_PREFIX
