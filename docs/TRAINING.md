# Training — list-price QLoRA

Config: [`training/configs/list_price_qlora.yaml`](../training/configs/list_price_qlora.yaml)

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
uv run modal run --detach src/priceengine/training/qlora_job.py \
  --config training/configs/list_price_qlora.yaml

# Existing Modal weights still live under the old folder name until you retrain:
uv run priceengine eval --modal \
  --adapter-path /data/checkpoints/amazon_replica \
  --name list_price_qlora \
  --limit 100 \
  --out reports/leaderboard.md
```

Eval needs `HF_TOKEN` (local tokenizer) + Modal `pricer-service` for the
published baseline, or `--no-include-baseline` / cached `reports/leaderboard.json`.

## Related

- [`COMPARISON.md`](COMPARISON.md) — fair eval
- [`DESIGN.md`](DESIGN.md) — architecture
