"""Modal QLoRA training job (A100 / A10G).

Flow (match this when reading the file)::

    spawn_training_from_laptop   # local entrypoint ``main``
      └─ run_qlora_on_a100 / run_qlora_on_a10g
           └─ run_qlora_training
                1. maybe_start_wandb
                2. load_quantized_base_model
                3. load_train_and_val_prompts   # Hub first, else Modal volume
                4. build_lora_adapter_config
                5. build_sft_trainer_args
                6. SFTTrainer.train → save adapter under /data/checkpoints/{name}

::

    modal run --detach src/priceengine/train/modal_train.py \\
      --config configs/qlora.yaml

Default data: Hub ``benifa/items_prompts_full``.
Fallback: ``dataset.local_dir`` (default ``/data/hf_dataset``) from
``priceengine build-local-sft``.
Requires Modal secret ``huggingface-secret`` (``HF_TOKEN``).
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

huggingface_secret = [modal.Secret.from_name("huggingface-secret")]
training_volume = modal.Volume.from_name("price-engine-data", create_if_missing=True)
VOLUME_MOUNT = "/data"
PROMPT_COMPLETION_COLUMNS = ["prompt", "completion"]


# ---------------------------------------------------------------------------
# Step helpers (called from run_qlora_training in order)
# ---------------------------------------------------------------------------


def maybe_start_wandb(train_config: dict[str, Any]) -> None:
    """Step 1 — optional W&B run (needs WANDB_API_KEY in the Modal environment)."""
    import os
    from datetime import UTC, datetime

    if not train_config.get("wandb", {}).get("enabled"):
        return
    if not os.environ.get("WANDB_API_KEY"):
        return

    import wandb

    os.environ.setdefault("WANDB_PROJECT", "price-engine")
    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        name=f"{train_config['name']}-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}",
        config=train_config,
        tags=train_config.get("wandb", {}).get("tags", []),
    )


def load_quantized_base_model(train_config: dict[str, Any]):
    """Step 2 — load frozen 4-bit base LM + tokenizer (the “Q” in QLoRA)."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    quant_cfg = train_config["quantization"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
    )

    base_model_id = train_config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=quantization_config, device_map="auto"
    )
    base_model.config.use_cache = False
    return base_model, tokenizer


def load_prompts_from_hub(
    hub_dataset_id: str, *, val_split_name: str, val_row_limit: int | None
) -> tuple[Any, Any]:
    """Load ``prompt`` / ``completion`` columns from a Hub dataset."""
    from datasets import load_dataset

    dataset = load_dataset(hub_dataset_id)
    train_prompts = dataset["train"].select_columns(PROMPT_COMPLETION_COLUMNS)
    if val_split_name not in dataset:
        return train_prompts, None
    val_prompts = dataset[val_split_name].select_columns(PROMPT_COMPLETION_COLUMNS)
    if val_row_limit:
        val_prompts = val_prompts.select(
            range(min(int(val_row_limit), len(val_prompts)))
        )
    return train_prompts, val_prompts


def load_prompts_from_local_volume(
    local_dataset_dir: str, *, val_split_name: str, val_row_limit: int | None
) -> tuple[Any, Any]:
    """Load prompts from Modal volume path (output of ``build-local-sft``)."""
    from datasets import load_from_disk

    dataset = load_from_disk(local_dataset_dir)
    train_prompts = dataset["train"].select_columns(PROMPT_COMPLETION_COLUMNS)
    # Hub uses "val"; build-local-sft uses "validation".
    for split_name in (val_split_name, "validation", "val"):
        if split_name in dataset:
            val_prompts = dataset[split_name].select_columns(PROMPT_COMPLETION_COLUMNS)
            if val_row_limit:
                val_prompts = val_prompts.select(
                    range(min(int(val_row_limit), len(val_prompts)))
                )
            return train_prompts, val_prompts
    return train_prompts, None


def load_train_and_val_prompts(train_config: dict[str, Any]) -> tuple[Any, Any]:
    """Step 3 — Hub prompts first; fall back to volume ``hf_dataset``."""
    dataset_cfg = train_config.get("dataset") or {}
    hub_dataset_id = dataset_cfg.get("hub_id")
    local_dataset_dir = dataset_cfg.get("local_dir") or f"{VOLUME_MOUNT}/hf_dataset"
    val_split_name = dataset_cfg.get("val_split", "val")
    val_row_limit = dataset_cfg.get("val_size")

    if hub_dataset_id:
        try:
            return load_prompts_from_hub(
                hub_dataset_id,
                val_split_name=val_split_name,
                val_row_limit=val_row_limit,
            )
        except Exception as exc:  # noqa: BLE001 — intentional Hub → local fallback
            print(
                f"Hub dataset {hub_dataset_id!r} unavailable ({exc}); "
                f"falling back to local {local_dataset_dir}"
            )

    return load_prompts_from_local_volume(
        local_dataset_dir,
        val_split_name=val_split_name,
        val_row_limit=val_row_limit,
    )


def build_lora_adapter_config(train_config: dict[str, Any]):
    """Step 4 — PEFT LoRA config from YAML ``lora:`` block."""
    from peft import LoraConfig

    lora = train_config["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        task_type=lora["task_type"],
        target_modules=lora["target_modules"],
    )


def build_sft_trainer_args(
    train_config: dict[str, Any],
    *,
    adapter_checkpoint_dir: str,
    has_validation_split: bool,
    use_early_stopping: bool,
    early_stop_metric: str,
) -> Any:
    """Step 5 — TRL ``SFTConfig`` from YAML ``training:`` / ``token_budget:``."""
    from trl import SFTConfig

    training = train_config["training"]
    token_budget = train_config["token_budget"]
    trainer_kwargs: dict[str, Any] = dict(
        output_dir=adapter_checkpoint_dir,
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
        eval_strategy="steps" if has_validation_split else "no",
        eval_steps=training["eval_steps"] if has_validation_split else None,
        save_total_limit=int(training.get("save_total_limit", 10)),
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model=early_stop_metric if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        max_length=token_budget["max_seq_length"],
        report_to=["wandb"] if train_config.get("wandb", {}).get("enabled") else [],
        seed=training["seed"],
        # Loss only on the completion (the dollar amount), not the question text.
        completion_only_loss=bool(training.get("completion_only", True)),
    )
    # Newer TRL dropped group_by_length from SFTConfig (Colab used an older TRL).
    if "group_by_length" in inspect.signature(SFTConfig.__init__).parameters:
        trainer_kwargs["group_by_length"] = bool(
            training.get("group_by_length", False)
        )
    return SFTConfig(**trainer_kwargs)


def save_adapter_checkpoint(
    trainer: Any, tokenizer: Any, adapter_checkpoint_dir: str
) -> None:
    """Persist LoRA adapter + tokenizer and commit the Modal volume."""
    trainer.model.save_pretrained(adapter_checkpoint_dir)
    tokenizer.save_pretrained(adapter_checkpoint_dir)
    training_volume.commit()


def run_qlora_training(config_yaml: str) -> str:
    """Orchestrate steps 1–6; return adapter path or Hub model id."""
    import yaml
    from transformers import EarlyStoppingCallback, set_seed
    from trl import SFTTrainer

    train_config = yaml.safe_load(config_yaml)
    set_seed(int(train_config["training"]["seed"]))

    # 1
    maybe_start_wandb(train_config)
    # 2
    base_model, tokenizer = load_quantized_base_model(train_config)
    # 3
    train_prompts, val_prompts = load_train_and_val_prompts(train_config)

    early_stop_cfg = train_config.get("early_stop") or {}
    use_early_stopping = bool(early_stop_cfg.get("enabled")) and val_prompts is not None
    early_stop_metric = early_stop_cfg.get("metric", "eval_loss")
    if use_early_stopping and early_stop_metric not in ("eval_loss", "loss"):
        raise ValueError(
            f"early_stop.metric={early_stop_metric!r} is unsupported; use eval_loss "
            "(Trainer does not emit MAE)."
        )

    adapter_checkpoint_dir = f"{VOLUME_MOUNT}/checkpoints/{train_config['name']}"
    # 4 + 5
    lora_adapter_config = build_lora_adapter_config(train_config)
    trainer_args = build_sft_trainer_args(
        train_config,
        adapter_checkpoint_dir=adapter_checkpoint_dir,
        has_validation_split=val_prompts is not None,
        use_early_stopping=use_early_stopping,
        early_stop_metric=early_stop_metric,
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stop_cfg.get("patience", 3))
            )
        )

    trainer = SFTTrainer(
        model=base_model,
        args=trainer_args,
        train_dataset=train_prompts,
        eval_dataset=val_prompts,
        peft_config=lora_adapter_config,
        processing_class=tokenizer,
        callbacks=callbacks or None,
    )
    # 6
    trainer.train()
    save_adapter_checkpoint(trainer, tokenizer, adapter_checkpoint_dir)

    hub_model_id = train_config.get("hub_model_id")
    if hub_model_id:
        trainer.model.push_to_hub(hub_model_id, private=True)
        return hub_model_id
    return adapter_checkpoint_dir


# ---------------------------------------------------------------------------
# Modal GPU entrypoints + laptop spawn
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    secrets=huggingface_secret,
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT: training_volume},
)
def run_qlora_on_a100(config_yaml: str) -> str:
    """Modal A100 worker — selected when ``modal.gpu`` starts with A100."""
    return run_qlora_training(config_yaml)


@app.function(
    image=image,
    secrets=huggingface_secret,
    gpu="A10G",
    timeout=12 * 60 * 60,
    volumes={VOLUME_MOUNT: training_volume},
)
def run_qlora_on_a10g(config_yaml: str) -> str:
    """Modal A10G worker — selected for non-A100 ``modal.gpu`` values."""
    return run_qlora_training(config_yaml)


@app.local_entrypoint()
def main(
    config: str = "configs/qlora.yaml",
    wait: bool = False,
):
    """Laptop entry: read YAML, spawn the matching Modal GPU function."""
    import yaml

    config_yaml = Path(config).read_text()
    train_config = yaml.safe_load(config_yaml)
    gpu_name = (train_config.get("modal") or {}).get("gpu", "A100-40GB")
    modal_worker = (
        run_qlora_on_a100
        if str(gpu_name).upper().startswith("A100")
        else run_qlora_on_a10g
    )

    if wait:
        adapter_path = modal_worker.remote(config_yaml)
        print(f"Training complete: {adapter_path}")
        return

    job = modal_worker.spawn(config_yaml)
    print(f"Training spawned on {gpu_name}: {job.object_id}")
    print(f"Dashboard: {job.get_dashboard_url()}")
    print(f"Adapter → /data/checkpoints/{train_config.get('name', 'run')}")
    print("Use --detach with fire-and-forget or the spawn dies when local exits.")
