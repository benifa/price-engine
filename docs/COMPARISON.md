# Comparison protocol vs `ed-donner/price-2025-11-28`

## Fairness rules

1. **Same base:** `meta-llama/Llama-3.2-3B`.
2. **Same serve path:** 4-bit NF4, double quant, bf16 compute, `set_seed(42)`, `max_new_tokens=5`, regex after `Price is $`.
3. **Native prompts:** list-price format (`amazon_list`) for Amazon checkpoints.
4. **Same golden items** for every model (`data/golden/amazon.parquet`).
5. **Paired bootstrap** on per-item absolute errors for ΔMAE (10k resamples).

## Eval sets

| Name | Data | Purpose |
|------|------|---------|
| `amazon` | Lite test (`benifa/items_lite` → golden) | Headline claim |
| `baselines` | Same golden, CPU baselines | Sanity floor |

## Metrics

- MAE ($), median APE, hit-rate (error &lt; $40 **or** &lt; 20% of truth), RMSLE.

## Victory

`relative_improvement = (mae_baseline - mae_challenger) / mae_baseline ≥ 0.25` **and** paired CI lower bound &gt; 0.

## Models

| ID | Adapter |
|----|---------|
| Published baseline | Amazon adapter via Modal `pricer-service` |
| list_price_qlora | Our list-price QLoRA adapter |
| Same-category train median | Guess median train price in that category |
| Overall train median | Guess one number: median of all train prices |

Implementation: `src/priceengine/eval/`.
