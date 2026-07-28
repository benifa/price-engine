"""Pricer protocol and local model adapters for the fair leaderboard.

Anything that can turn an ``EvalItem`` into a dollar guess implements ``Pricer``.
The CLI / leaderboard never care *how* the guess was produced — only ``name``
and ``price(item)``.

Adapters here
-------------
* ``SameCategoryMedianPricer`` / ``OverallMedianPricer`` — CPU sanity floors
  (guess a training-set median; no neural net)
* ``FineTunedPricer`` — local LoRA (or full) causal LM completing ``Price is $``

Remote adapters live in ``published_baseline`` (published checkpoint) and
``adapter_scoring`` (our Modal batch job).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Protocol, runtime_checkable

from priceengine.models import EvalItem

logger = logging.getLogger(__name__)


def extract_price(text: str) -> float:
    """Parse the first number from model output after ``Price is $``.

    Returns ``0.0`` if nothing parses — treated as a failed generation, which
    correctly hurts MAE so silent parse failures do not look like wins.
    """
    text = text.replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    return float(match.group()) if match else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


@runtime_checkable
class Pricer(Protocol):
    """Minimal scoring interface used by ``leaderboard.run_pricer``."""

    name: str

    def price(self, item: EvalItem) -> float: ...


# Leaderboard labels — plain English so outsiders can read the table.
SAME_CATEGORY_MEDIAN_NAME = "Same-category train median"
OVERALL_TRAIN_MEDIAN_NAME = "Overall train median"


class SameCategoryMedianPricer:
    """Guess the median train price among items in the *same category*.

    Example: for an Electronics product, predict the median of all Electronics
    prices seen in training. Falls back to the overall train median if the
    category is unseen. A real model should beat this floor.
    """

    def __init__(
        self,
        train_items: list[EvalItem],
        name: str = SAME_CATEGORY_MEDIAN_NAME,
    ):
        self.name = name
        by_category: dict[str, list[float]] = defaultdict(list)
        all_prices: list[float] = []
        for item in train_items:
            all_prices.append(item.price)
            if item.category:
                by_category[item.category].append(item.price)
        self._overall_median = _median(all_prices)
        self._category_medians = {
            category: _median(prices) for category, prices in by_category.items()
        }

    def price(self, item: EvalItem) -> float:
        if item.category and item.category in self._category_medians:
            return self._category_medians[item.category]
        return self._overall_median


class OverallMedianPricer:
    """Always guess one number: the median price across the whole train set.

    Ignores the product text entirely — the weakest useful baseline.
    """

    def __init__(self, value: float, name: str = OVERALL_TRAIN_MEDIAN_NAME):
        self.value = value
        self.name = name

    def price(self, item: EvalItem) -> float:
        return self.value


# Back-compat aliases (older imports / notebooks).
CategoryMedianPricer = SameCategoryMedianPricer
ConstantPricer = OverallMedianPricer

class FineTunedPricer:
    """Local Hugging Face LoRA that completes the list-price prompt.

    Generation knobs match the published baseline (seed 42, ``max_new_tokens=5``,
    parse after ``Price is $``) so local macOS smoke tests are comparable to
    Modal scores — see ``docs/COMPARISON.md``.

    On macOS (MPS/CPU) pass ``load_in_4bit=False``; bitsandbytes needs CUDA.
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_model: str | None = None,
        revision: str | None = None,
        name: str | None = None,
        max_new_tokens: int = 5,
        cutoff_tokens: int | None = None,
        load_in_4bit: bool = True,
    ):
        from transformers import AutoTokenizer, set_seed

        from priceengine.config import BASE_MODEL, SUMMARY_CUTOFF

        base_model = base_model or BASE_MODEL
        self.name = name or model_id
        self.max_new_tokens = max_new_tokens
        self.cutoff_tokens = SUMMARY_CUTOFF if cutoff_tokens is None else cutoff_tokens
        set_seed(42)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = self._load_peft_model(
            base_model, model_id, revision=revision, load_in_4bit=load_in_4bit
        )
        logger.info("Loaded fine-tuned pricer %s (4bit=%s)", model_id, load_in_4bit)

    @staticmethod
    def _load_peft_model(
        base_model: str,
        adapter_id: str,
        *,
        revision: str | None,
        load_in_4bit: bool,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=quant, device_map="auto"
            )
        else:
            dtype = (
                torch.float16 if torch.backends.mps.is_available() else torch.float32
            )
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            base = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=dtype, device_map=None
            )
            base.to(device)

        model = PeftModel.from_pretrained(base, adapter_id, revision=revision)
        model.eval()
        return model

    def price(self, item: EvalItem) -> float:
        import torch
        from transformers import set_seed

        from priceengine.config import PRICE_PREFIX
        from priceengine.training.prompts import prompt_for_eval_item, truncate_text

        # Re-seed every call so item order does not change generations (fair eval).
        set_seed(42)
        body, _ = truncate_text(
            item.text_for_pricing(), self.tokenizer, self.cutoff_tokens
        )
        prompt = prompt_for_eval_item(item, text=body)
        device = next(self.model.parameters()).device
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, max_new_tokens=self.max_new_tokens
            )
        decoded = self.tokenizer.decode(outputs[0])
        if PRICE_PREFIX in decoded:
            decoded = decoded.split(PRICE_PREFIX, 1)[1]
        return extract_price(decoded)
