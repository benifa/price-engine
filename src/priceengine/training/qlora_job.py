"""Modal QLoRA training job (A100 / A10G).

Replica of Ed Donner week-7 Colab (``LITE_MODE=False``)::

    modal run --detach src/priceengine/training/qlora_job.py \\
      --config training/configs/list_price_qlora.yaml

Data
----
By default the YAML points at Hub ``ed-donner/items_prompts_full`` (already
prompt/completion shaped). If ``dataset.hub_id`` is null, the job falls back to
``/data/hf_dataset`` on the Modal volume (from ``build-sft-dataset``).

Requires Modal secret ``huggingface-secret`` (``HF_TOKEN``). A100 recommended
for batch size 256. Adapter lands at ``/data/checkpoints/{config.name}/``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import modal

app = modal.App("price-engine-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.8.0",
        "transformers>=4.46",
        "peft>=0.14",
        "trl>=0.20",
        "datasets>=3.0",
        "bitsandbytes>=0.44",
        "accelerate>=1.0",
        "wandb>=0.18",
        "pyyaml>=6.0",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

secrets = [modal.Secret.from_name("huggingface-secret")]
volume = modal.Volume.from_name("price-engine-data", create_if_missing=True)
DATA = "/data"
_PROMPT_COMPLETION_COLUMNS = ["prompt", "completion"]


def _maybe_init_wandb(cfg: dict[str, Any]) -> None:
    import os
    from datetime import UTC, datetime

    if not cfg.get("wandb", {}).get("enabled"):
        return
    if not os.environ.get("WANDB_API_KEY"):
        return

    import wandb

    os.environ.setdefault("WANDB_PROJECT", "price-engine")
    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        name=f"{cfg['name']}-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}",
        config=cfg,
        tags=cfg.get("wandb", {}).get("tags", []),
    )


def _load_base_model(cfg: dict[str, Any]):
    """4-bit base LM + tokenizer; cache disabled for gradient checkpointing / LoRA."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    quant_cfg = cfg["quantization"]
    quant = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
    )

    base_id = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=quant, device_map="auto"
    )
    model.config.use_cache = False
    return model, tokenizer


def _lora_config(cfg: dict[str, Any]):
    from peft import LoraConfig

    lora = cfg["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        task_type=lora["task_type"],
        target_modules=lora["target_modules"],
    )


def _load_prompt_completion_splits(
    cfg: dict[str, Any], dataset_dir: str
) -> tuple[Any, Any]:
    """Return ``(train_ds, val_ds_or_None)`` with only prompt/completion columns."""
    ds_cfg = cfg.get("dataset") or {}
    hub_id = ds_cfg.get("hub_id")

    if hub_id:
        from datasets import load_dataset

        dataset = load_dataset(hub_id)
        train_ds = dataset["train"].select_columns(_PROMPT_COMPLETION_COLUMNS)
        val_key = ds_cfg.get("val_split", "val")
        if val_key not in dataset:
            return train_ds, None
        val_ds = dataset[val_key].select_columns(_PROMPT_COMPLETION_COLUMNS)
        val_size = ds_cfg.get("val_size")
        if val_size:
            val_ds = val_ds.select(range(min(int(val_size), len(val_ds))))
        return train_ds, val_ds

    from datasets import load_from_disk

    dataset = load_from_disk(dataset_dir)
    train_ds = dataset["train"].select_columns(_PROMPT_COMPLETION_COLUMNS)
    if "val" in dataset:
        return train_ds, dataset["val"].select_columns(_PROMPT_COMPLETION_COLUMNS)
    if "validation" in dataset:
        return train_ds, dataset["validation"].select_columns(
            _PROMPT_COMPLETION_COLUMNS
        )
    return train_ds, None


def _build_sft_config(
    cfg: dict[str, Any],
    *,
    output_dir: str,
    has_val: bool,
    early_stop: bool,
    early_metric: str,
) -> Any:
    """TRL ``SFTConfig`` from YAML; tolerates older/newer TRL signatures."""
    from trl import SFTConfig

    training = cfg["training"]
    token_budget = cfg["token_budget"]
    sft_kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=int(
            training.get("per_device_eval_batch_size", 1)
        ),
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        lr_scheduler_type=training["lr_scheduler_type"],
        warmup_ratio=training["warmup_ratio"],
        optim=training["optim"],
        max_grad_norm=training["max_grad_norm"],
        weight_decay=training["weight_decay"],
        bf16=training["bf16"],
        fp16=training["fp16"],
        logging_steps=training["logging_steps"],
        save_steps=training["save_steps"],
        eval_strategy="steps" if has_val else "no",
        eval_steps=training["eval_steps"] if has_val else None,
        save_total_limit=int(training.get("save_total_limit", 10)),
        load_best_model_at_end=early_stop,
        metric_for_best_model=early_metric if early_stop else None,
        greater_is_better=False if early_stop else None,
        max_length=token_budget["max_seq_length"],
        report_to=["wandb"] if cfg.get("wandb", {}).get("enabled") else [],
        seed=training["seed"],
        # Loss only on the completion (the dollar amount), not the question text.
        completion_only_loss=bool(training.get("completion_only", True)),
    )
    # Newer TRL dropped group_by_length from SFTConfig (Colab used an older TRL).
    if "group_by_length" in inspect.signature(SFTConfig.__init__).parameters:
        sft_kwargs["group_by_length"] = bool(training.get("group_by_length", False))
    return SFTConfig(**sft_kwargs)


def _run_training(config_yaml: str, dataset_dir: str) -> str:
    """Fine-tune with QLoRA per YAML; return adapter path or Hub model id."""
    import yaml
    from transformers import EarlyStoppingCallback, set_seed
    from trl import SFTTrainer

    cfg = yaml.safe_load(config_yaml)
    set_seed(int(cfg["training"]["seed"]))
    _maybe_init_wandb(cfg)

    model, tokenizer = _load_base_model(cfg)
    train_ds, val_ds = _load_prompt_completion_splits(cfg, dataset_dir)

    early_cfg = cfg.get("early_stop") or {}
    early_stop = bool(early_cfg.get("enabled")) and val_ds is not None
    early_metric = early_cfg.get("metric", "eval_loss")
    if early_stop and early_metric not in ("eval_loss", "loss"):
        raise ValueError(
            f"early_stop.metric={early_metric!r} is unsupported; use eval_loss "
            "(Trainer does not emit MAE)."
        )

    output_dir = f"{DATA}/checkpoints/{cfg['name']}"
    sft_args = _build_sft_config(
        cfg,
        output_dir=output_dir,
        has_val=val_ds is not None,
        early_stop=early_stop,
        early_metric=early_metric,
    )

    callbacks = []
    if early_stop:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_cfg.get("patience", 3))
            )
        )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=_lora_config(cfg),
        processing_class=tokenizer,
        callbacks=callbacks or None,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    volume.commit()

    hub_model_id = cfg.get("hub_model_id")
    if hub_model_id:
        trainer.model.push_to_hub(hub_model_id, private=True)
        return hub_model_id
    return output_dir


@app.function(
    image=image,
    secrets=secrets,
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    volumes={DATA: volume},
)
def train_a100(config_yaml: str, dataset_dir: str = f"{DATA}/hf_dataset") -> str:
    return _run_training(config_yaml, dataset_dir)


@app.function(
    image=image,
    secrets=secrets,
    gpu="A10G",
    timeout=12 * 60 * 60,
    volumes={DATA: volume},
)
def train_a10g(config_yaml: str, dataset_dir: str = f"{DATA}/hf_dataset") -> str:
    return _run_training(config_yaml, dataset_dir)


@app.local_entrypoint()
def main(
    config: str = "training/configs/list_price_qlora.yaml",
    wait: bool = False,
):
    """Spawn training on Modal. GPU comes from ``modal.gpu`` in the YAML (default A100)."""
    import yaml

    yaml_text = Path(config).read_text()
    cfg = yaml.safe_load(yaml_text)
    gpu = (cfg.get("modal") or {}).get("gpu", "A100-40GB")
    runner = train_a100 if str(gpu).upper().startswith("A100") else train_a10g

    if wait:
        result = runner.remote(yaml_text)
        print(f"Training complete: {result}")
        return

    call = runner.spawn(yaml_text)
    print(f"Training spawned on {gpu}: {call.object_id}")
    print(f"Dashboard: {call.get_dashboard_url()}")
    print(f"Adapter → /data/checkpoints/{cfg.get('name', 'run')}")
    print("Use --detach with fire-and-forget or the spawn dies when local exits.")
