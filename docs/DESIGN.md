# Design — Price Engine

## Mission

Ship a **description → market value** engine trained on **realized sold prices**, with calibrated confidence and comparable evidence, and prove it beats `ed-donner/price-2025-11-28` on used-goods valuation under a fair protocol.

Deal Hunter and other apps consume this engine via API; they are not this repo’s core.

## Architecture

1. **Corpus** — Apify eBay solds (+ Browse API tracker) → clean → time-split train/val/test.
2. **Training** — QLoRA on Llama-3.2-3B; YAML configs; W&B; completion-only loss after `Price is $`.
3. **Eval** — `Pricer` protocol; Ed + ours + baselines; MAE / hit-rate / paired bootstrap CIs.
4. **Serving** — Modal: specialist (+ sold-comps RAG / ensemble).

See the README ablation ladder (R0–R4) for how wins are attributed.

## Data

- Prices in **$1–$999**, rounded (single-token integers in Llama).
- Condition enum: `new | open-box | refurb | used-good | used-fair | for-parts`.
- Splits by **sold_date** (not random): train &lt; T−14d, val mid week, test freshest week.
- Near-duplicate titles+condition dropped across split boundaries.
- RAG store built from train+val only.

## Training (vs Ed)

| Axis | Ed | Ours |
|------|----|------|
| Labels | Amazon list prices (new) | eBay sold prices (any condition) |
| Prompt | description only | + condition / category / sold month |
| LoRA targets | attention (R1 control) | attention + MLP (R2) |
| Stopping | 1 epoch | early stop on **val MAE** (R2+) |
| Quant | 4-bit NF4 double quant | same at train **and** serve |

Hyperparameters for each run live in `training/configs/*.yaml`.

## Evaluation victory

On the used-goods golden set vs Ed (R0):

- Relative MAE improvement ≥ 25%, **and**
- 95% paired-bootstrap CI on ΔMAE excludes 0, **and**
- Beat category-median and zero-shot GPT.

`items_lite` results are published for integrity even if Ed wins there.

## Moat

The outcome tracker accumulates longitudinal sold/unsold resolutions that cannot be bought retroactively. Retrain monthly; re-grade weekly on new out-of-time sales.

## Out of scope (this repo)

Telegram bots, Deal Hunter planners, affiliate links — those stay in the app repo.
