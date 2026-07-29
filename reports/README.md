# Reports (generated locally — gitignored)

Artifacts from the research loop: **prepare → Modal/HF train → eval → publish**.
Ollama export stages files under `artifacts/` (not here).

| Artifact | Produced by |
|----------|-------------|
| `amazon_prep.json` | `priceengine prepare-list-prices` |
| `sft_dataset.json` | `priceengine build-sft-dataset` |
| `leaderboard*.md` / `.json` | `priceengine eval` / `eval-baselines` |
| `eval_report.html` | `priceengine visualize-eval` / `eval --visualize` |
| `eval_report-v*.html` | Versioned copy when `--version` / `--report-version` is set |
| `publish-v*.json` | `priceengine publish-model` (Hub PEFT tag metadata) |
| `token_length/` | `priceengine token-budget` |

Regenerate from the CLI — nothing here is committed.
