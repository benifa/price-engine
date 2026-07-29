# Comparison protocol vs `ed-donner/price-2025-11-28`

Fair eval of our **Modal/HF QLoRA adapter** against the published baseline on the
**same** Amazon golden items. Visual HTML (`eval --visualize`) is an optional
report of the same predictions — it does not change the victory math.

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
| list_price_qlora | Our list-price QLoRA adapter (Modal train; Hub tag when published) |
| Same-category train median | Guess median train price in that category |
| Overall train median | Guess one number: median of all train prices |

## Reports

| Artifact | Command |
|----------|---------|
| `reports/leaderboard.md` + `.json` | `priceengine eval` |
| `reports/eval_report.html` (overlay, worst misses) | `eval --visualize` or `visualize-eval` |
| Versioned HTML | `--report-version v0.1.0` → `eval_report-v0.1.0.html` |

Implementation: `src/priceengine/eval/`. Publish tagged adapters after a clean
eval: [`PUBLISH.md`](PUBLISH.md).
