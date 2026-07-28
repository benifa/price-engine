"""Prepare Amazon list-price datasets on disk.

Public surface used by the CLI, training, and eval:

* ``prepare_dataset`` — Hub → local splits + golden (see ``dataset_builder``)
* ``load_products`` / ``load_eval_items`` — read those artifacts (see ``parquet_records``)
"""

from priceengine.data_prep.dataset_builder import (
    clamp_usd_price,
    hub_row_to_product,
    prepare_dataset,
)
from priceengine.data_prep.parquet_records import (
    load_eval_items,
    load_products,
    products_to_eval_items,
    save_eval_items,
    save_json,
    save_products,
)

__all__ = [
    "clamp_usd_price",
    "hub_row_to_product",
    "load_eval_items",
    "load_products",
    "prepare_dataset",
    "products_to_eval_items",
    "save_eval_items",
    "save_json",
    "save_products",
]
