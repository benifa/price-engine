"""QLoRA training for Amazon list-price estimation.

Pipeline
--------
1. ``prompts`` — list-price prompt / completion text (shared with eval).
2. ``token_budget`` — pick description CUTOFF from train-split histograms.
3. ``sft_dataset`` — optional: local parquet splits → HF prompt/completion set.
4. ``qlora_job`` — Modal A100/A10G job; usually loads Hub
   ``ed-donner/items_prompts_full`` (see ``training/configs/list_price_qlora.yaml``).

Typical path (replica)
----------------------
Local golden + Modal Hub train::

    priceengine prepare-list-prices --size lite
    modal run --detach src/priceengine/training/qlora_job.py \\
      --config training/configs/list_price_qlora.yaml

Then score with ``priceengine eval --modal --adapter-path /data/checkpoints/...``.

Loss: completion-only after ``Price is $`` so the model learns the dollar amount,
not the question text.
"""
