"""Publish a versioned LoRA adapter to the Hugging Face Hub.

Other apps load the adapter with PEFT against ``BASE_MODEL`` and the list-price
prompt format (see ``docs/PUBLISH.md``).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from priceengine.config import AMAZON_LIST_QUESTION, BASE_MODEL, PRICE_PREFIX, Settings

logger = logging.getLogger(__name__)

DEFAULT_HUB_REPO = "benifa/list-price-qlora"
_TAG_RE = re.compile(r"^v?\d+\.\d+\.\d+([.-].+)?$|^20\d{2}-\d{2}-\d{2}$")


def normalize_tag(tag: str) -> str:
    """Accept ``v0.1.0`` or ``2026-07-28``; ensure a leading ``v`` for semver."""
    tag = tag.strip()
    if not tag:
        raise ValueError("tag must be non-empty")
    if re.fullmatch(r"\d+\.\d+\.\d+([.-].+)?", tag):
        tag = f"v{tag}"
    if not _TAG_RE.fullmatch(tag):
        raise ValueError(f"tag {tag!r} must look like v0.1.0 or 2026-07-28")
    return tag


def _metrics_blurb(leaderboard_md: Path | None) -> str:
    if not leaderboard_md or not leaderboard_md.exists():
        return "_No leaderboard.md found — run `priceengine eval` before publishing._\n"
    return leaderboard_md.read_text().strip() + "\n"


def build_model_card(
    *,
    repo_id: str,
    tag: str,
    base_model: str,
    adapter_path: Path,
    leaderboard_md: Path | None,
    private: bool,
) -> str:
    """Markdown model card for Hub consumers."""
    visibility = "private" if private else "public"
    return f"""---
language:
- en
library_name: peft
base_model: {base_model}
tags:
- qlora
- peft
- pricing
- list-price
- llama
license: mit
---

# {repo_id} (`{tag}`)

QLoRA adapter for Amazon **list-price** estimation. Trained and evaluated by
[price-engine](https://github.com/benifa/price-engine).

| Field | Value |
|-------|--------|
| Base model | `{base_model}` |
| Adapter source | `{adapter_path}` |
| Revision tag | `{tag}` |
| Visibility | {visibility} |
| Published (UTC) | {datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} |

## Prompt format

```text
{AMAZON_LIST_QUESTION}

<title + description>

{PRICE_PREFIX}
```

The model completes with `NNN.00` (integer dollars in $1–$999).

## Load in another app

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

base = "{base_model}"
adapter = "{repo_id}"  # revision="{tag}"

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base, quantization_config=quant, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter, revision="{tag}")
```

## Eval snapshot

{_metrics_blurb(leaderboard_md)}

## Ollama

This Hub repo is a **PEFT adapter**, not a GGUF. See `docs/OLLAMA.md` to merge
and convert for local Ollama use.
"""


def publish_adapter(
    adapter_path: Path,
    *,
    repo_id: str = DEFAULT_HUB_REPO,
    tag: str,
    private: bool = True,
    base_model: str = BASE_MODEL,
    leaderboard_md: Path | None = None,
    settings: Settings | None = None,
) -> dict[str, str]:
    """Upload adapter files to Hub and create a revision tag.

    Returns ``{"repo_id", "tag", "url"}``.
    """
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import HfHubHTTPError

    tag = normalize_tag(tag)
    adapter_path = adapter_path.expanduser().resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    if not (adapter_path / "adapter_config.json").exists():
        logger.warning(
            "No adapter_config.json in %s — uploading directory anyway",
            adapter_path,
        )

    api = HfApi()
    create_repo(repo_id, private=private, exist_ok=True, repo_type="model")

    card = build_model_card(
        repo_id=repo_id,
        tag=tag,
        base_model=base_model,
        adapter_path=adapter_path,
        leaderboard_md=leaderboard_md,
        private=private,
    )
    meta = {
        "repo_id": repo_id,
        "tag": tag,
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "private": private,
        "published_utc": datetime.now(UTC).isoformat(),
    }

    with tempfile.TemporaryDirectory(prefix="priceengine-publish-") as tmp:
        staging = Path(tmp) / "upload"
        shutil.copytree(
            adapter_path,
            staging,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
        )
        (staging / "README.md").write_text(card)
        (staging / "priceengine_publish.json").write_text(json.dumps(meta, indent=2))
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Publish adapter {tag}",
        )

    try:
        api.create_tag(
            repo_id=repo_id,
            tag=tag,
            tag_message=f"price-engine release {tag}",
            repo_type="model",
        )
    except HfHubHTTPError as exc:
        # Tag may already exist from a prior publish of the same version.
        logger.warning("create_tag(%s): %s", tag, exc)

    url = f"https://huggingface.co/{repo_id}/tree/{tag}"
    logger.info("Published %s @ %s → %s", repo_id, tag, url)

    if settings is not None:
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        out = settings.reports_dir / f"publish-{tag}.json"
        out.write_text(json.dumps({**meta, "url": url}, indent=2))
        logger.info("Wrote %s", out)

    return {"repo_id": repo_id, "tag": tag, "url": url}
