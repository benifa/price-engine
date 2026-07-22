# Comparison protocol vs `ed-donner/price-2025-11-28`

## Fairness rules

1. **Same base:** `meta-llama/Llama-3.2-3B`.
2. **Same serve path:** 4-bit NF4, double quant, bf16 compute, `set_seed(42)`, `max_new_tokens=5`, regex after `Price is $`.
3. **Native prompts:** Ed’s checkpoint is prompted with Ed’s week-7 format; ours with our conditioned format. Cross-prompting is not used for the headline table.
4. **Same golden items** for every contestant on a battleground.
5. **Paired bootstrap** on per-item absolute errors for ΔMAE (10k resamples).

## Battlegrounds

| Name | Data | Purpose |
|------|------|---------|
| `used_goods` | Time-split eBay solds (golden) | Headline claim |
| `items_lite` | Ed’s Amazon test split | Integrity / distribution shift |
| `used_goods_baselines` | Same golden, CPU baselines | Sanity floor |

## Metrics

- MAE ($), median APE, hit-rate (error &lt; $40 **or** &lt; 20% of truth), RMSLE.
- Optional: calibration when confidence is emitted.
- Slices: category × condition; truncated vs not.

## Victory

`relative_improvement = (mae_ed - mae_ours) / mae_ed ≥ 0.25` **and** paired CI lower bound &gt; 0.

## Contestants

| ID | Adapter |
|----|---------|
| R0 | `FineTunedPricer(ed-donner/..., prompt_style="ed")` |
| R1–R3 | Our adapters with `prompt_style="ours"` |
| baselines | `CategoryMedianPricer`, `ConstantPricer`, `ZeroShotFrontierPricer` |
| R4 | RAG / ensemble wrappers (post sold-comps store) |

Implementation: `src/priceengine/eval/`.
