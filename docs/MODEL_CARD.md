# Model card — Price Engine specialist (draft)

> Fill metrics after R1/R2 complete. Do not claim victory until `reports/leaderboard-used_goods.md` shows a passing paired comparison.

## Model details

- **Base:** `meta-llama/Llama-3.2-3B` (base, not Instruct)
- **Adaptation:** QLoRA (see `training/configs/`)
- **Training data:** eBay sold / completed listings (private corpus); prices $1–$999 rounded
- **Languages:** English product descriptions
- **License:** Code MIT; base model subject to Meta Llama license; do not redistribute scraped raw data

## Intended use

Estimate a **resale / sold** market value from a free-text description plus optional condition and category. For research, Deal Hunter pricing, and seller guidance — not as sole authority for high-stakes financial decisions.

## Out-of-scope

- Items without text descriptions
- Prices outside $1–$999 (excluded from training)
- Exact collectibles / one-of-ones with no comps
- Legal appraisal / insurance valuation without human review

## Evaluation data

- **Primary:** time-split held-out eBay solds (`data/golden/used_goods.parquet`, not in git)
- **Secondary:** `ed-donner/items_lite` test

## Metrics (to be filled)

| Contestant | Battleground | MAE | Hit rate | vs Ed ΔMAE (95% CI) |
|------------|--------------|-----|----------|---------------------|
| R0 Ed | used_goods | — | — | — |
| R1 | used_goods | — | — | — |
| R2 | used_goods | — | — | — |

## Ethical considerations

Training data is scraped marketplace content. We do not publish raw listings. Estimates can be wrong; confidence and comps should be shown to users. Automation that messages sellers must respect platform ToS.

## Citation

Ed Donner’s LLM engineering course week 7 established the QLoRA price-completion recipe and published `ed-donner/price-2025-11-28`. This project reuses that recipe with transactional targets and a stricter eval protocol.
