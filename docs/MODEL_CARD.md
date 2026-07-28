# Model card — Price Engine Amazon specialist (draft)

> Fill metrics after the list-price QLoRA eval completes. Do not claim victory until
> `reports/leaderboard.md` shows a passing paired comparison.

## Model details

- **Base:** `meta-llama/Llama-3.2-3B` (base, not Instruct)
- **Adaptation:** QLoRA (`training/configs/list_price_qlora.yaml`)
- **Training data:** Amazon list-price prompts (`ed-donner/items_prompts_full` on Modal; local mirrors `benifa/items_*`)
- **Languages:** English product descriptions
- **License:** Code MIT; base model subject to Meta Llama license

## Intended use

Estimate an Amazon-style list price from a free-text description. Research and
pricing tooling — not sole authority for high-stakes financial decisions.

## Out-of-scope

- Items without text descriptions
- Prices outside $1–$999
- Exact collectibles / one-of-ones with no comps

## Evaluation data

- **Primary:** Amazon lite test (`benifa/items_lite` → `data/golden/amazon.parquet`)

## Metrics (to be filled)

| Model | Eval set | MAE | Hit rate | vs baseline ΔMAE (95% CI) |
|-------|----------|-----|----------|---------------------------|
| Published baseline | amazon | — | — | — |
| list_price_qlora | amazon | — | — | — |

## Citation

The QLoRA price-completion recipe and published checkpoint
`ed-donner/price-2025-11-28` come from Ed Donner’s LLM engineering course week 7.
This project mirrors the Amazon list-price datasets under `benifa/` and reuses that
recipe with a stricter paired-bootstrap eval protocol.
