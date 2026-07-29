# Publish a versioned LoRA adapter to Hugging Face Hub

## Role in the pipeline

```text
Modal/HF QLoRA train  →  eval (+ HTML)  →  publish-model --tag vX.Y.Z  →  other apps
```

| | |
|--|--|
| **What you publish** | PEFT LoRA adapter + model card + Hub revision tag |
| **What other apps load** | `base_model` + `adapter_id` + `revision=` |
| **What this is not** | Ollama/GGUF upload (optional later — [`OLLAMA.md`](OLLAMA.md)) |

Training runtime remains **Modal + Hugging Face**. Publishing does not retrain.

## Default repo

`benifa/list-price-qlora`

## One command

```bash
# After eval (so reports/leaderboard.md can feed the model card):
uv run priceengine publish-model \
  --adapter-path /path/to/adapter \
  --repo benifa/list-price-qlora \
  --tag v0.1.0 \
  --private
```

Use `--public` to create/update a public model repo. Tags accept semver
(`v0.1.0` / `0.1.0`) or a date (`2026-07-28`).

Requires `HF_TOKEN` with write access to the repo.

## What gets uploaded

| Artifact | Purpose |
|----------|---------|
| LoRA weights + `adapter_config.json` | PEFT load |
| `README.md` | Model card (prompt format, load snippet, eval snapshot) |
| `priceengine_publish.json` | Machine-readable publish metadata |
| Hub revision tag | Pin consumers to `revision="v0.1.0"` |

In-training push remains optional via `hub_model_id` in
[`list_price_qlora.yaml`](../training/configs/list_price_qlora.yaml). Prefer
CLI publish so you can attach metrics and an explicit tag after eval.

## Consumer contract

| Field | Value |
|-------|--------|
| `base_model` | `meta-llama/Llama-3.2-3B` |
| `adapter_id` | `benifa/list-price-qlora` |
| `revision` | e.g. `v0.1.0` |
| Prompt | question + product text + `Price is $` (see model card) |

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

base = "meta-llama/Llama-3.2-3B"
adapter = "benifa/list-price-qlora"
revision = "v0.1.0"

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base, quantization_config=quant, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter, revision=revision)
```

## Versioned eval HTML

```bash
uv run priceengine eval --modal ... --visualize --report-version v0.1.0
# → reports/eval_report-v0.1.0.html (+ reports/eval_report.html)
```

Link that report from the Hub model card or release notes when you publish.
