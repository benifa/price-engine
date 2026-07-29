# Price Engine

Fine-tune and evaluate a model that predicts a product **list price** from its
title and description. Then publish the weights to Hugging Face.

1. Prepare data  
2. Fine-tune (QLoRA on Modal)  
3. Evaluate vs a published baseline  
4. Publish a versioned Hub model  

Data today: Amazon list prices ($1–$999).

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env   # HF_TOKEN=...

uv run priceengine prepare-list-prices --size lite

uv run modal run --detach src/priceengine/training/qlora_job.py \
  --config training/configs/list_price_qlora.yaml

uv run priceengine eval --modal \
  --adapter-path /data/checkpoints/amazon_replica \
  --name list_price_qlora --limit 100 --visualize \
  --out reports/leaderboard.md

uv run priceengine publish-model \
  --adapter-path ./adapters/list_price_qlora \
  --repo benifa/list-price-qlora --tag v0.1.0 --private
```

Needs `HF_TOKEN` + Modal. See `uv run priceengine --help`.

## Prompt

```text
What does this cost to the nearest dollar?

<title + description>

Price is $
```

## Docs

[`DESIGN`](docs/DESIGN.md) · [`TRAINING`](docs/TRAINING.md) · [`COMPARISON`](docs/COMPARISON.md) ·
[`PUBLISH`](docs/PUBLISH.md) · [`OLLAMA`](docs/OLLAMA.md) · [`MODEL_CARD`](docs/MODEL_CARD.md)

## License

MIT
