# Data (local only — gitignored)

| Path | Contents |
|------|----------|
| `raw/` | Apify dumps (JSONL) |
| `clean/` | Normalized parquet after cleaning |
| `splits/` | `train.parquet`, `val.parquet`, `test.parquet` |
| `golden/` | Held-out golden set used for the public leaderboard |
| `vectorstore/` | Sold-comps Chroma store (train+val only) |

Nothing under `data/` is committed except this README. Training corpora and golden blobs live in private HF datasets; adapters live in a private HF model repo.
