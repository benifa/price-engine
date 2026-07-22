"""Modal QLoRA training entrypoint.

Deploy / run:
    modal run src/priceengine/training/modal_train.py --config training/configs/ed_replica.yaml

Requires Modal secret `price-engine-secrets` with HF_TOKEN and WANDB_API_KEY,
and the prepared dataset at data/hf_dataset (or a private HF dataset id in the config).
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("price-engine-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.8.0",
        "transformers>=4.46",
        "peft>=0.14",
        "trl>=0.12",
        "datasets>=3.0",
        "bitsandbytes>=0.44",
        "accelerate>=1.0",
        "wandb>=0.18",
        "pyyaml>=6.0",
        "scikit-learn>=1.5",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

secrets = [modal.Secret.from_name("price-engine-secrets")]
volume = modal.Volume.from_name("price-engine-data", create_if_missing=True)
DATA = "/data"


def _load_config(path: str) -> dict:
    import yaml

    with open(path) as handle:
        return yaml.safe_load(handle)


@app.function(
    image=image,
    secrets=secrets,
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={DATA: volume},
)
def train(config_yaml: str, dataset_dir: str = f"{DATA}/hf_dataset") -> str:
    """Fine-tune with QLoRA per YAML config; return HF adapter repo id or local path."""
    import os
    from datetime import datetime

    import torch
    import yaml
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    cfg = yaml.safe_load(config_yaml)
    set_seed(int(cfg["training"]["seed"]))

    if cfg.get("wandb", {}).get("enabled") and os.environ.get("WANDB_API_KEY"):
        import wandb

        os.environ.setdefault("WANDB_PROJECT", "price-engine")
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=f"{cfg['name']}-{datetime.utcnow().strftime('%Y%m%d-%H%M')}",
            config=cfg,
            tags=cfg.get("wandb", {}).get("tags", []),
        )

    q = cfg["quantization"]
    quant = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
    )

    base = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=quant, device_map="auto"
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"],
    )

    ds = load_from_disk(dataset_dir)

    def to_text(example):
        return {"text": example["prompt"] + example["completion"]}

    train_ds = ds["train"].map(to_text)
    val_ds = ds["validation"].map(to_text) if "validation" in ds else None

    t = cfg["training"]
    tb = cfg["token_budget"]
    output_dir = f"{DATA}/checkpoints/{cfg['name']}"
    sft = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        optim=t["optim"],
        max_grad_norm=t["max_grad_norm"],
        weight_decay=t["weight_decay"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=t["eval_steps"] if val_ds is not None else None,
        max_length=tb["max_seq_length"],
        report_to=["wandb"] if cfg.get("wandb", {}).get("enabled") else [],
        seed=t["seed"],
        dataset_text_field="text",
    )

    collator = None
    if t.get("completion_only"):
        collator = DataCollatorForCompletionOnlyLM(
            response_template=t["response_template"], tokenizer=tokenizer
        )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    volume.commit()

    hub_id = cfg.get("hub_model_id")
    if hub_id:
        trainer.model.push_to_hub(hub_id, private=True)
        return hub_id
    return output_dir


@app.local_entrypoint()
def main(config: str = "training/configs/ed_replica.yaml"):
    path = Path(config)
    yaml_text = path.read_text()
    result = train.remote(yaml_text)
    print(f"Training complete: {result}")
