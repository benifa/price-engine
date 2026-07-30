# Data (local only — gitignored)

| Path | Contents |
|------|----------|
| `splits/` | `train.parquet`, `val.parquet`, `test.parquet` (from `prepare-data`) |
| `golden/` | `amazon.parquet` held-out eval set |
| `hf_dataset/` | Optional prompt/completion DatasetDict from `build-local-sft` (Hub-prompts fallback) |

Nothing under `data/` is committed except this README.
