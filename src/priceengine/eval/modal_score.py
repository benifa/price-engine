"""Modal batch scoring for our LoRA adapter on the price-engine-data volume.

Used by ``run.score_challenger_on_modal`` when you pass ``priceengine eval --modal``.

Loads ``/data/checkpoints/...`` and completes prompts the caller already built
(question + product text + ``Price is $``).

::

    modal run src/priceengine/eval/modal_score.py \\
        --adapter-path /data/checkpoints/list_price_qlora

Keep ``BASE_MODEL`` / ``PRICE_PREFIX`` in sync with ``priceengine.config`` —
this file runs on Modal and does not import the local package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

app = modal.App("price-engine-eval-adapter")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.8.0",
        "transformers>=4.46",
        "peft>=0.14",
        "bitsandbytes>=0.44",
        "accelerate>=1.0",
        "sentencepiece",
        "protobuf",
    )
)

huggingface_secret = [modal.Secret.from_name("huggingface-secret")]
training_volume = modal.Volume.from_name("price-engine-data", create_if_missing=False)
VOLUME_MOUNT = "/data"

# Keep in sync with priceengine.config — package is not on the Modal image.
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PRICE_PREFIX = "Price is $"
MAX_NEW_TOKENS = 5


def parse_price_from_completion(decoded: str) -> float:
    """Read the first number after ``Price is $`` (same idea as ``pricers.extract_price``)."""
    contents = (
        decoded.split(PRICE_PREFIX, 1)[1] if PRICE_PREFIX in decoded else decoded
    )
    contents = contents.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0.0


@app.function(
    image=image,
    secrets=huggingface_secret,
    gpu="T4",
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: training_volume},
)
def score_prompts_with_adapter(
    prompts: list[str],
    adapter_path: str = f"{VOLUME_MOUNT}/checkpoints/list_price_qlora",
) -> list[float]:
    """Load QLoRA once; return one dollar estimate per full prompt."""
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quantization_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    estimates: list[float] = []
    for index, prompt in enumerate(prompts):
        # Same seed every item — generation must not depend on batch order.
        set_seed(42)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
        estimates.append(parse_price_from_completion(tokenizer.decode(outputs[0])))
        if (index + 1) % 25 == 0:
            print(f"priced {index + 1}/{len(prompts)}")
    return estimates


@app.local_entrypoint()
def main(
    prompts_json: str = "data/eval_prompts.json",
    adapter_path: str = "/data/checkpoints/list_price_qlora",
    out: str = "data/adapter_preds.json",
):
    """Optional offline entry: JSON list of prompts → JSON list of dollar estimates."""
    prompts = json.loads(Path(prompts_json).read_text())
    estimates = score_prompts_with_adapter.remote(prompts, adapter_path)
    Path(out).write_text(json.dumps(estimates))
    print(f"Wrote {len(estimates)} predictions → {out}")
