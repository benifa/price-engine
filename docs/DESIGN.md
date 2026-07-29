# Design — Price Engine

## Mission

Reproduce and improve on the published Amazon list-price QLoRA specialist
(`ed-donner/price-2025-11-28`) with a **fair**, paired-bootstrap eval on a held-out
Amazon golden set, then publish a **versioned** Hub adapter other apps can load.

This repo is a closed research loop:

**prepare data → train LoRA on Modal/HF → eval (+ HTML) → tag + publish to Hub**

It is not a marketplace scraper, a production serving API, or an Ollama fine-tune
stack. Ollama appears only as an optional GGUF export ([`OLLAMA.md`](OLLAMA.md)).

## Stack boundaries

| Layer | Technology | Artifact |
|-------|------------|----------|
| Data | Hugging Face datasets | `data/splits/`, `data/golden/` |
| Train | Modal + Transformers/PEFT QLoRA | `/data/checkpoints/list_price_qlora` |
| Eval | Local + Modal scoring | `reports/leaderboard.*`, `reports/eval_report*.html` |
| Share | Hub PEFT repo + revision tag | e.g. `benifa/list-price-qlora@v0.1.0` |
| Optional local | Merge + GGUF → Ollama | Not the training source of truth |

## System overview

```mermaid
flowchart TB
  subgraph sources["Sources"]
    Lite["benifa/items_lite"]
    Full["benifa/items_full"]
    Prompts["ed-donner/items_prompts_full"]
    Baseline["ed-donner/price-2025-11-28<br/>via Modal pricer-service"]
  end

  subgraph local["Local disk"]
    Splits["data/splits/*.parquet"]
    Golden["data/golden/amazon.parquet"]
    Reports["reports/leaderboard.* + eval_report*.html"]
  end

  subgraph modal["Modal — train & score"]
    TrainJob["qlora_job<br/>A100 / A10G"]
    Adapter["/data/checkpoints/{name}"]
    ScoreJob["adapter_scoring<br/>T4 batch"]
  end

  subgraph hub["Hugging Face Hub — share"]
    Tagged["tagged PEFT adapter<br/>publish-model"]
  end

  Lite --> Splits
  Lite --> Golden
  Full -.->|optional size=full| Splits
  Prompts --> TrainJob
  TrainJob --> Adapter
  Golden --> ScoreJob
  Adapter --> ScoreJob
  Baseline --> Reports
  ScoreJob --> Reports
  Splits -.->|CPU medians| Reports
  Adapter --> Tagged
```

## Package boundaries

Cross-cutting types and knobs live at the package root so `data_prep`, `training`,
and `eval` do not depend on each other sideways:

| Module | Owns |
|--------|------|
| `cli.py` | User commands only (thin orchestration) |
| `config.py` | Paths, `$1–$999` bounds, prompt constants, victory thresholds |
| `models.py` | `ProductListing`, `EvalItem`, `Prediction`, metrics DTOs |
| `data_prep/` | Hub download → typed parquet on disk |
| `training/` | Prompt text, CUTOFF, Modal train, Hub publish, Ollama export stub |
| `eval/` | Pricers, metrics, leaderboard, Modal score, Plotly HTML reports |

```mermaid
flowchart LR
  subgraph shared["Shared root"]
    C[config]
    M[models]
  end

  DP[data_prep] --> C
  DP --> M
  TR[training] --> C
  TR --> M
  EV[eval] --> C
  EV --> M

  TR -.->|prompts reused| EV
```

`training.prompts` is the single source of prompt/completion text. Eval imports it
so train and score never drift on format.

## End-to-end data flow

```mermaid
sequenceDiagram
  participant Hub as Hugging Face Hub
  participant Prep as data_prep
  participant Disk as data/
  participant Modal as Modal GPU
  participant Eval as eval
  participant Out as reports/
  participant Pub as publish-model

  Hub->>Prep: items_lite / items_full
  Prep->>Disk: splits + golden parquet
  Hub->>Modal: items_prompts_full (train)
  Modal->>Modal: QLoRA → adapter on volume
  Disk->>Eval: golden EvalItems
  Modal->>Eval: adapter guesses (or local FineTunedPricer)
  Hub->>Eval: published baseline via pricer-service
  Eval->>Out: leaderboard.md + .json + optional HTML
  Note over Pub: after eval (metrics on card)
  Pub->>Hub: tagged PEFT adapter (v0.1.0)
```

### Stage 1 — Data prep

- Hub rows → `ProductListing` (`list_price`, `item_id`, …).
- Prices clamped/rounded to **[$1, $999]** (Llama encodes those integers as one token).
- **lite:** keep publisher train/val/test; golden = lite test.
- **full:** subsample large train/val; golden still = lite test; those titles are
  **held out** of train/val so eval cannot leak.

### Stage 2 — Training (Modal + Hugging Face only)

- **Runtime:** Modal GPU; base `meta-llama/Llama-3.2-3B`; PEFT LoRA. **Not Ollama.**
- Prompt format: question + product text + `Price is $` → completion `NNN.00`.
- Default replica loads Hub prompt/completion rows inside Modal (no local SFT build).
- Optional local path: `build-sft-dataset` from parquet splits → `data/hf_dataset/`.
- Loss is **completion-only** after the prefix (learn dollars, not the question).
- New runs write `/data/checkpoints/list_price_qlora` (older volume path:
  `amazon_replica`). Details: [`TRAINING.md`](TRAINING.md).

### Stage 3 — Eval (+ visual report)

Anything that maps `EvalItem → float` implements the `Pricer` protocol:

| Adapter | Where it runs |
|---------|----------------|
| `SameCategoryMedianPricer` / `OverallMedianPricer` | Local CPU (train medians) |
| `FineTunedPricer` | Local (macOS: no 4-bit) |
| `PublishedBaselinePricer` | Modal `pricer-service` |
| `adapter_scoring.price_batch` | Modal T4, our adapter on the volume |

`priceengine eval --visualize` (or `visualize-eval`) writes Plotly HTML: MAE bars,
challenger-vs-baseline overlay, per-model scatter / running MAE, and a worst-misses
table. Version with `--report-version v0.1.0` → `reports/eval_report-v0.1.0.html`.

Metrics and victory rules: [`COMPARISON.md`](COMPARISON.md).

### Stage 4 — Versioned Hub publish

- CLI: `priceengine publish-model --adapter-path … --tag v0.1.0 [--public|--private]`.
- Uploads PEFT adapter + model card (prompt contract, load snippet, leaderboard
  snapshot) and creates a Hub revision tag other apps pin with `revision=`.
- Prefer this over YAML `hub_model_id` (untagged one-shot push). See [`PUBLISH.md`](PUBLISH.md).

### Optional — Ollama export

Merge LoRA → fp16 → GGUF → `ollama create`. Documented in [`OLLAMA.md`](OLLAMA.md).
The tagged Hub **PEFT** adapter remains the source of truth.

## Prompt contract

```text
What does this cost to the nearest dollar?

{title}
{description}

Price is $
```

Description tokens are truncated to `SUMMARY_CUTOFF` (default **110**) before
wrapping so the question + prefix always fit. Training `max_seq_length` for the
replica is **128**.

**Published baseline caveat:** the remote `Pricer` wraps the description itself.
`PublishedBaselinePricer` therefore sends a truncated description only — not a
full prompt — to avoid doubling the question.

## On-disk layout

```text
data/
  splits/{train,val,test}.parquet   # ProductListing
  golden/amazon.parquet             # EvalItem (fair-eval set)
  combined/amazon.parquet           # optional union dump
  hf_dataset/                       # optional local SFT DatasetDict
reports/
  leaderboard*.md / .json
  eval_report.html / eval_report-v*.html
  publish-v*.json                   # publish-model metadata
  amazon_prep.json
  token_length/                     # CUTOFF histograms
```

Nothing under `data/` or generated `reports/` is committed except READMEs.

## Design decisions

| Decision | Why |
|----------|-----|
| List prices, not sold comps | Matches the published specialist and Hub datasets |
| `$1–$999` rounded labels | Single-token integers in Llama; stable completions |
| Completion-only loss | Model capacity spent on the dollar amount |
| Same golden for every model | Unpaired eval would not support a fair ΔMAE claim |
| Paired bootstrap + 25% bar | Wins must be large *and* statistically one-sided |
| Modal for 4-bit train/score | bitsandbytes is awkward on macOS; course baseline already on Modal |
| Shared `prompts` module | Train/eval format cannot silently diverge |
| Hub PEFT + tags for consumers | Other apps pin `revision=vX.Y.Z`; see [`PUBLISH.md`](PUBLISH.md) |
| Ollama = export only | Fine-tune stays Modal/HF LoRA; GGUF is optional ([`OLLAMA.md`](OLLAMA.md)) |

## Out of scope

- Marketplace scrapers / Apify / eBay sold comps
- RAG over comparable listings
- Production HTTP serving or online pricing APIs
- Native Ollama fine-tuning (export path only)
- Multi-prompt-format experiments (one format: `amazon_list`)

## Related docs

- [`COMPARISON.md`](COMPARISON.md) — fairness rules and victory definition
- [`TRAINING.md`](TRAINING.md) — exact Colab-full hyperparameters
- [`PUBLISH.md`](PUBLISH.md) — versioned Hub adapters for other apps
- [`OLLAMA.md`](OLLAMA.md) — optional GGUF / Ollama consumer path
- [`MODEL_CARD.md`](MODEL_CARD.md) — intended use and results table
