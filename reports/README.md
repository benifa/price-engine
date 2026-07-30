# Reports (local only — gitignored)

| Artifact | Command |
|----------|---------|
| `amazon_prep.json` | `priceengine prepare-data` |
| `sft_dataset.json` | `priceengine build-local-sft` (optional) |
| `leaderboard*.md` / `.json` | `priceengine eval` |
| `eval_report.html` / `eval_report-v*.html` | `eval --visualize` |
| `publish-v*.json` | `priceengine publish-model` |

Nothing under `reports/` is committed except this README.
