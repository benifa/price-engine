"""Published Amazon baseline scored via Modal ``pricer-service``.

Why Modal instead of loading the Hub adapter locally?
    bitsandbytes 4-bit weights are awkward on macOS; the course already deploys
    ``ed-donner/price-2025-11-28`` behind a Modal ``Pricer`` class. Calling that
    service keeps our eval path identical to the published serve path
    (``docs/COMPARISON.md``).

Contract with the remote class
------------------------------
We send a *truncated product description only*. The remote wrapper adds the
list-price question and ``Price is $`` prefix itself — do not send a full
prompt here or the model will see the question twice.
"""

from __future__ import annotations

import logging

from priceengine.models import EvalItem
from priceengine.training.prompts import truncate_text

logger = logging.getLogger(__name__)

BASELINE_NAME = "ed-donner/price-2025-11-28 (Modal)"


class PublishedBaselinePricer:
    """Published list-price adapter on Modal (``pricer-service`` / ``Pricer``)."""

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

        from priceengine.config import BASE_MODEL, SUMMARY_CUTOFF

        remote_cls = modal.Cls.from_name(modal_app, modal_class)
        self._remote = remote_cls()
        self._cutoff_tokens = (
            SUMMARY_CUTOFF if cutoff_tokens is None else cutoff_tokens
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
