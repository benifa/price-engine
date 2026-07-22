"""End-to-end corpus prep on a synthetic JSONL dump (no Apify / network)."""

import json
from datetime import date, timedelta
from pathlib import Path

from priceengine.config import Settings
from priceengine.corpus.apify_pull import load_raw_jsonl
from priceengine.corpus.cleaning import clean_listings, drop_log_outliers
from priceengine.corpus.io import save_eval_items, save_listings
from priceengine.corpus.splits import remove_boundary_near_dupes, time_split, to_eval_items


def test_synthetic_jsonl_pipeline(tmp_path: Path):
    as_of = date(2026, 7, 21)
    rows = []
    for i in range(40):
        sold = as_of - timedelta(days=30 - (i % 28))
        rows.append(
            {
                "itemId": f"id-{i}",
                "title": f"Test Product Model {i % 7} Widget",
                "description": (
                    "A detailed description of a consumer electronic device "
                    "with accessories included."
                ),
                "soldPrice": 50 + (i * 3) % 200,
                "soldDate": sold.isoformat(),
                "condition": "Used - Good",
                "category": "Electronics",
                "url": f"https://example.com/{i}",
            }
        )
    dump = tmp_path / "dump.jsonl"
    dump.write_text("\n".join(json.dumps(r) for r in rows))

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
    )
    raw = load_raw_jsonl(dump)
    cleaned, _ = clean_listings(raw)
    cleaned, _ = drop_log_outliers(cleaned)
    assert len(cleaned) >= 20

    save_listings(settings.clean_dir / "sold.parquet", cleaned)
    train, val, test = time_split(cleaned, as_of=as_of)
    train, val, test, _ = remove_boundary_near_dupes(train, val, test)
    save_listings(settings.splits_dir / "train.parquet", train)
    save_listings(settings.splits_dir / "val.parquet", val)
    save_listings(settings.splits_dir / "test.parquet", test)
    golden = to_eval_items(test)
    save_eval_items(settings.golden_dir / "used_goods.parquet", golden)

    assert (settings.splits_dir / "train.parquet").exists()
    assert len(golden) == len(test)
