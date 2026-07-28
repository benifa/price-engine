"""Fair evaluation of list-price models against a held-out Amazon golden set.

Flow (see also ``docs/COMPARISON.md``):

1. Load golden ``EvalItem`` rows (``data/golden/amazon.parquet``).
2. Score each model through the ``Pricer`` protocol → ``Prediction`` rows.
3. Summarize MAE / hit-rate / RMSLE (``metrics``) and write a leaderboard.
4. Optionally pair-bootstrap a challenger vs the published Modal baseline.

Modules
-------
pricers
    Local adapters: train-median baselines and a fine-tuned LoRA.
published_baseline
    Remote published checkpoint via Modal ``pricer-service``.
adapter_scoring
    Modal job that batch-completes prompts for *our* adapter on GPU.
metrics
    Aggregate stats + paired bootstrap victory test.
leaderboard
    Orchestrate scoring and write ``reports/leaderboard*.md``.
visualization
    Course-style Plotly HTML (truth vs guess + running MAE).
"""
