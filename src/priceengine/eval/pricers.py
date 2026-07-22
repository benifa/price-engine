"""Pricer protocol and adapters for the fair leaderboard."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Protocol, runtime_checkable

from priceengine.models import EvalItem
from priceengine.training.prompts import prompt_for_eval_item

logger = logging.getLogger(__name__)


def extract_price(text: str) -> float:
    text = text.replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    return float(match.group()) if match else 0.0


@runtime_checkable
class Pricer(Protocol):
    name: str

    def price(self, item: EvalItem) -> float: ...


class CategoryMedianPricer:
    """Dumb baseline: predict the training-set median price for the item's category."""

    def __init__(self, train_items: list[EvalItem], name: str = "category-median"):
        self.name = name
        buckets: dict[str, list[float]] = defaultdict(list)
        all_prices: list[float] = []
        for item in train_items:
            all_prices.append(item.price)
            if item.category:
                buckets[item.category].append(item.price)
        self._global = float(sorted(all_prices)[len(all_prices) // 2]) if all_prices else 0.0
        self._medians = {
            cat: float(sorted(prices)[len(prices) // 2]) for cat, prices in buckets.items()
        }

    def price(self, item: EvalItem) -> float:
        if item.category and item.category in self._medians:
            return self._medians[item.category]
        return self._global


class ConstantPricer:
    def __init__(self, value: float, name: str | None = None):
        self.value = value
        self.name = name or f"constant-{value:.0f}"

    def price(self, item: EvalItem) -> float:
        return self.value


class ZeroShotFrontierPricer:
    """Frontier LLM with no retrieval — prompt-only estimate."""

    def __init__(self, model: str = "gpt-4.1-mini", name: str | None = None):
        from openai import OpenAI

        self.model = model
        self.name = name or f"zero-shot-{model}"
        self.client = OpenAI()

    def price(self, item: EvalItem) -> float:
        prompt = (
            "Estimate the resale price of this product in USD. "
            "Respond with only a number.\n\n"
            f"{item.text_for_pricing()}"
        )
        if item.condition:
            prompt = f"Condition: {item.condition}\n{prompt}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            seed=42,
        )
        return extract_price(response.choices[0].message.content or "0")


class FineTunedPricer:
    """Loads a HF LoRA (or full) causal LM and completes `Price is $`.

    Used for both Ed's published adapter and ours. `prompt_style` must match
    the format the checkpoint was trained on (`ed` vs `ours`) — grading Ed
    with our header would sandbag him.
    """

    def __init__(
        self,
        model_id: str,
        *,
        prompt_style: str,
        base_model: str = "meta-llama/Llama-3.2-3B",
        revision: str | None = None,
        name: str | None = None,
        max_new_tokens: int = 5,
        cutoff_tokens: int = 110,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

        self.name = name or model_id
        self.prompt_style = prompt_style
        self.max_new_tokens = max_new_tokens
        self.cutoff_tokens = cutoff_tokens
        set_seed(42)

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, model_id, revision=revision)
        self.model.eval()
        logger.info("Loaded fine-tuned pricer %s (style=%s)", model_id, prompt_style)

    def price(self, item: EvalItem) -> float:
        import torch
        from transformers import set_seed

        from priceengine.training.prompts import truncate_text

        set_seed(42)
        body, _ = truncate_text(item.text_for_pricing(), self.tokenizer, self.cutoff_tokens)
        prompt = prompt_for_eval_item(item, style=self.prompt_style, text=body)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=self.max_new_tokens)
        decoded = self.tokenizer.decode(outputs[0])
        # Prefer text after the price prefix if present
        from priceengine.config import PRICE_PREFIX

        if PRICE_PREFIX in decoded:
            decoded = decoded.split(PRICE_PREFIX, 1)[1]
        return extract_price(decoded)
