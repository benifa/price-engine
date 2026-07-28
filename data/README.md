# Data (local only — gitignored)

| Path | Contents |
|------|----------|
| `combined/` | `amazon.parquet` union of prepared splits |
| `splits/` | `train.parquet`, `val.parquet`, `test.parquet` |
| `golden/` | `amazon.parquet` held-out eval set |
| `hf_dataset/` | Optional local prompt/completion dataset from `build-sft-dataset` |

Nothing under `data/` is committed except this README.
