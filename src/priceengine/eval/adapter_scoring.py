"""Modal batch scoring for *our* LoRA adapter on the ``price-engine-data`` volume.

Unlike ``published_baseline`` (which calls the course ``pricer-service``), this
job loads an adapter from ``/data/checkpoints/...`` and completes prompts that
the *caller already built* (question + text + ``Price is $``).

    # Standalone smoke test
    modal run src/priceengine/eval/adapter_scoring.py \\
        --adapter-path /data/checkpoints/list_price_qlora

The Modal image does **not** install this package, so model id / prefix strings
are duplicated as literals below (keep in sync with ``priceengine.config``).
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

secrets = [modal.Secret.from_name("huggingface-secret")]
volume = modal.Volume.from_name("price-engine-data", create_if_missing=False)
DATA = "/data"

# Keep in sync with priceengine.config — package is not on the Modal image.
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PRICE_PREFIX = "Price is $"
MAX_NEW_TOKENS = 5


def _parse_completion(decoded: str) -> float:
    """Mirror ``pricers.extract_price`` after stripping ``Price is $``."""
    contents = (
        decoded.split(PRICE_PREFIX, 1)[1] if PRICE_PREFIX in decoded else decoded
    )
    contents = contents.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0.0


@app.function(
    image=image,
    secrets=secrets,
    gpu="T4",
    timeout=60 * 60,
    volumes={DATA: volume},
)
def price_batch(
    prompts: list[str],
    adapter_path: str = f"{DATA}/checkpoints/list_price_qlora",
) -> list[float]:
    """Load QLoRA once; complete each full prompt (already ends with ``Price is $``)."""
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    guesses: list[float] = []
    for index, prompt in enumerate(prompts):
        # Same seed every item — generation must not depend on batch order.
        set_seed(42)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
        guesses.append(_parse_completion(tokenizer.decode(outputs[0])))
        if (index + 1) % 25 == 0:
            print(f"priced {index + 1}/{len(prompts)}")
    return guesses


@app.local_entrypoint()
def main(
    prompts_json: str = "data/eval_prompts.json",
    adapter_path: str = "/data/checkpoints/list_price_qlora",
    out: str = "data/adapter_preds.json",
):
    """Optional offline entry: JSON list of prompts → JSON list of guesses."""
    prompts = json.loads(Path(prompts_json).read_text())
    guesses = price_batch.remote(prompts, adapter_path)
    Path(out).write_text(json.dumps(guesses))
    print(f"Wrote {len(guesses)} predictions → {out}")
