"""Apify client for eBay sold / completed listings bootstrap."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from priceengine.config import Settings
from priceengine.corpus.cleaning import listing_from_apify_item
from priceengine.models import SoldListing

logger = logging.getLogger(__name__)

# Well-known Apify actors that expose eBay search / completed listings.
# Override with --actor if your account uses a different one.
DEFAULT_ACTOR = "dtrungtin/ebay-items-scraper"


def pull_sold_listings(
    settings: Settings,
    *,
    queries: list[str],
    max_items: int = 50_000,
    actor_id: str = DEFAULT_ACTOR,
    sold_only: bool = True,
) -> list[SoldListing]:
    """Run an Apify actor and parse items into SoldListing records.

    Requires APIFY_API_TOKEN. Writes the raw dataset JSONL under data/raw/.
    """
    if not settings.apify_token:
        raise RuntimeError(
            "APIFY_API_TOKEN is not set. Copy .env.example → .env and add your Apify token."
        )

    from apify_client import ApifyClient

    client = ApifyClient(settings.apify_token)
    run_input = {
        "queries": queries,
        "maxItems": max_items,
        "soldItems": sold_only,
        "proxyConfiguration": {"useApifyProxy": True},
    }
    logger.info(
        "Starting Apify actor %s for %d queries (max %d)",
        actor_id,
        len(queries),
        max_items,
    )
    run = client.actor(actor_id).call(run_input=run_input)
    dataset_id = run["defaultDatasetId"]

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = settings.raw_dir / f"apify_{dataset_id}.jsonl"
    listings: list[SoldListing] = []
    with raw_path.open("w") as out:
        for item in client.dataset(dataset_id).iterate_items():
            out.write(json.dumps(item) + "\n")
            parsed = listing_from_apify_item(item)
            if parsed is not None:
                listings.append(parsed)
    logger.info(
        "Parsed %d listings from Apify dataset %s → %s",
        len(listings),
        dataset_id,
        raw_path,
    )
    return listings


def load_raw_jsonl(path: Path) -> list[SoldListing]:
    """Reload a previously saved Apify dump without re-running the actor."""
    listings: list[SoldListing] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = listing_from_apify_item(json.loads(line))
            if parsed is not None:
                listings.append(parsed)
    return listings


# Default search queries covering our target categories (sold/completed inventory).
DEFAULT_QUERIES = [
    "laptop",
    "smartphone",
    "tablet",
    "headphones",
    "smart tv",
    "graphics card",
    "gaming console",
    "power tools",
    "robot vacuum",
    "bluetooth speaker",
    "monitor 4k",
    "dslr camera",
]
