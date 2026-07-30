"""Local pricers for the leaderboard.

Anything with ``name`` + ``price(item) -> float`` can be scored.
Median baselines live here; Modal adapters are in ``baseline_pricer`` / ``modal_score``.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Protocol, runtime_checkable

from priceengine.models import EvalItem

logger = logging.getLogger(__name__)

# Plain-English floors: "always guess a train median" — beat these before claiming wins.
SAME_CATEGORY_MEDIAN_NAME = "Naive floor · category median"
OVERALL_TRAIN_MEDIAN_NAME = "Naive floor · always train median"


def extract_price(text: str) -> float:
    """Parse the first number from model output. Returns 0.0 if nothing parses."""
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
    name: str

    def price(self, item: EvalItem) -> float: ...


class SameCategoryMedianPricer:
    """Guess the median train price in the same category (fallback: overall median)."""

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
    """Always guess one number: the median of all train prices."""

    def __init__(self, value: float, name: str = OVERALL_TRAIN_MEDIAN_NAME):
        self.value = value
        self.name = name

    def price(self, item: EvalItem) -> float:
        return self.value


class FineTunedPricer:
    """Local LoRA that completes the list-price prompt (macOS: load_in_4bit=False)."""

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

        from priceengine.config import BASE_MODEL, MAX_DESCRIPTION_TOKENS

        base_model = base_model or BASE_MODEL
        self.name = name or model_id
        self.max_new_tokens = max_new_tokens
        self.cutoff_tokens = (
            MAX_DESCRIPTION_TOKENS if cutoff_tokens is None else cutoff_tokens
        )
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
        from priceengine.prompts import prompt_for_eval_item, truncate_text

        set_seed(42)  # same seed each call so item order does not change outputs
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
