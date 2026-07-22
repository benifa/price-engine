# Price Engine

Calibrated **secondhand valuation from a bare description** — a QLoRA specialist trained on real sold prices, evaluated with a fair protocol against [`ed-donner/price-2025-11-28`](https://huggingface.co/ed-donner/price-2025-11-28_18.47.07).

> **Headline claim (used-goods golden set):** our specialist beats Ed’s published adapter by ≥25% relative MAE, with a 95% paired-bootstrap CI on ΔMAE that excludes zero. Results land in [`reports/`](reports/) as they are produced.

## Why this exists

Price-history tools need a clean product ID. Generic LLM guesses have no trained price structure. Ed’s course model (which we still ship in [deal-hunter-agent](https://github.com/benifa/deal-hunter-agent)) is strong on **Amazon list prices of new items**. This repo asks a different question:

> *Given a messy description and condition, what did this actually sell for?*

Same QLoRA recipe as week 7 — pointed at transactional data, with condition/time conditioning, ablation-tracked training, and a leaderboard that grades Ed as contestant zero.

## Repo map

| Path | Role |
|------|------|
| `src/priceengine/corpus/` | Apify pull, cleaning, time splits, leakage controls |
| `src/priceengine/training/` | Prompts, token budget, dataset prep, Modal QLoRA |
| `src/priceengine/eval/` | Pricer protocol, metrics, bootstrap, leaderboard |
| `src/priceengine/serving/` | Valuation API (Modal) |
| `training/configs/` | `ed_replica.yaml` (R1), `all_linear_r32.yaml` (R2), `all_linear_r64.yaml` (R3) |
| `docs/DESIGN.md` | Full architecture & decisions |
| `docs/COMPARISON.md` | Fair-eval protocol vs Ed |
| `docs/MODEL_CARD.md` | Intended use, limits, metrics |

Raw data and adapters are **not** committed (see `data/README.md`).

## Quickstart (Ed / items_lite path — current default)

No Apify required. Uses [`ed-donner/items_lite`](https://huggingface.co/datasets/ed-donner/items_lite) and Ed’s Modal specialist as R0.

```bash
uv sync --extra dev
cp .env.example .env   # HF_TOKEN, WANDB_API_KEY; OPENAI optional

# 1) Materialize Ed's dataset into our splits + golden set
uv run priceengine prepare-items-lite

# 2) Token budget + SFT dataset in Ed's prompt format
uv run priceengine token-budget
uv run priceengine prep-dataset --style ed --cutoff 110

# 3) CPU baselines on items_lite test
uv run priceengine eval-baselines

# 4) R0: Ed's published model via Modal (needs pricer-service deployed)
uv run priceengine eval-ed --limit 100

# 5) Train our control run on items_lite (same knobs as Ed)
modal run src/priceengine/training/modal_train.py \
  --config training/configs/items_lite_ed_format.yaml
```

Sold-listings (Apify) path remains available via `pull-apify` when you are ready.

## Ablation ladder (must stay in this order)

| Run | What changes | Question it answers |
|-----|----------------|---------------------|
| **R0** | Ed’s published checkpoint | Baseline on our golden set |
| **R1** | Our data + prompt; Ed’s exact knobs | Is the win from **data**? |
| **R2** | All-linear LoRA + MAE early stop | Is the win from **method**? |
| **R3** | r=64 | Is the win from **capacity**? |
| **R4** | Sold-comps RAG + re-fit ensemble | System-level accuracy |

Ed is always graded with **his** prompt format; we use **ours**. Same 4-bit serve path, same seed, same regex.

## Docs

- [Design](docs/DESIGN.md)
- [Comparison protocol](docs/COMPARISON.md)
- [Model card](docs/MODEL_CARD.md)

## License

MIT
