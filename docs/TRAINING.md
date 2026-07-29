# Training — list-price QLoRA

Config: [`training/configs/list_price_qlora.yaml`](../training/configs/list_price_qlora.yaml)

## What trains what

| | |
|--|--|
| **Trainer** | Modal GPU job (`qlora_job.py`) |
| **Base model** | Hugging Face `meta-llama/Llama-3.2-3B` |
| **Method** | QLoRA / PEFT LoRA |
| **Data** | Hugging Face `ed-donner/items_prompts_full` (loaded inside Modal) |
| **Output** | LoRA adapter on the Modal volume |
| **Not used** | Ollama fine-tune (see [`OLLAMA.md`](OLLAMA.md) for optional export only) |

After train → eval (+ HTML) → **versioned Hub publish** ([`PUBLISH.md`](PUBLISH.md)).
Prefer `priceengine publish-model --tag vX.Y.Z` over YAML `hub_model_id`.

## Exact recipe (full mode)

| Knob | Value |
|------|--------|
| Dataset | `ed-donner/items_prompts_full` (800k / 10k / 10k) — Hub load inside Modal |
| Local dataset mirrors | `benifa/items_lite`, `benifa/items_full` |
| Base | `meta-llama/Llama-3.2-3B` |
| Epochs | **3** |
| Batch size | **256** (grad accum 1) |
| Max seq length | **128** |
| LoRA r / α | **256 / 512** |
| Targets | attention **+** MLP (`q/k/v/o` + `gate/up/down`) |
| Dropout | 0.1 |
| LR / warmup / schedule | 1e-4 / 0.01 / cosine |
| Weight decay | 0.001 |
| Optim | `paged_adamw_32bit` |
| Quant | 4-bit NF4 double quant |
| `group_by_length` | true |
| Val size | 1000 |

## Run

```bash
# Train (Modal + HF) — writes /data/checkpoints/list_price_qlora
uv run modal run --detach src/priceengine/training/qlora_job.py \
  --config training/configs/list_price_qlora.yaml

# Eval + browser report
# Older volume folder still named amazon_replica — label it with --name:
uv run priceengine eval --modal \
  --adapter-path /data/checkpoints/amazon_replica \
  --name list_price_qlora \
  --limit 100 \
  --visualize --report-version v0.1.0 \
  --out reports/leaderboard.md

# Publish tagged Hub adapter for other apps (PEFT, not GGUF)
uv run priceengine publish-model \
  --adapter-path ./path/to/downloaded/adapter \
  --tag v0.1.0 --private
```

`hub_model_id` in the YAML is optional (one-shot private push, **no** revision tag /
metrics card). Prefer `publish-model` after eval.

Eval needs `HF_TOKEN` (local tokenizer) + Modal `pricer-service` for the
published baseline, or `--no-include-baseline` / cached `reports/leaderboard.json`.

## Related

- [`DESIGN.md`](DESIGN.md) — architecture and stack boundaries
- [`COMPARISON.md`](COMPARISON.md) — fair eval
- [`PUBLISH.md`](PUBLISH.md) — versioned Hub adapters
- [`OLLAMA.md`](OLLAMA.md) — optional GGUF / Ollama path (not the trainer)
