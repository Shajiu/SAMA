from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from swift import get_model_processor, get_template
from swift.trainers import Seq2SeqTrainingArguments

from sama.collator import SAMAMultiViewCollator
from sama.attn_capture import enable_last_layer_attention
from sama.config import build_structural_token_list, load_config
from sama.data import SAMADataset
from sama.modeling import (
    get_structural_token_ids,
    attach_structural_embedding_bank,
    count_trainable_parameters,
    freeze_backbone,
    replace_qkv_with_script_lora,
)
from sama.trainer import SAMASeq2SeqTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sama_7b.yaml")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--init_trainable_checkpoint", type=str, default=None,
                        help="Optional trainable-only SAMA checkpoint to initialize from; optimizer state is not resumed.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cli = parse_args()
    cfg = load_config(cli.config)
    train_cfg = cfg["training"]
    output_dir = cli.output_dir or train_cfg["output_dir"]
    set_seed(int(train_cfg.get("seed", 42)))

    if not torch.cuda.is_available():
        raise RuntimeError("This reference training script expects CUDA GPUs.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # Swift loads the registered Qwen2.5-VL model and processor. Eager attention is
    # required for the reference implementation of answer-to-visual attention loss.
    model, processor = get_model_processor(
        cfg["model"]["model_name_or_path"],
        model_type="qwen2_5_vl",
        new_special_tokens=build_structural_token_list(cfg),
        torch_dtype=torch.bfloat16 if cfg["model"].get("torch_dtype", "bfloat16") == "bfloat16" else torch.float16,
        attn_impl=cfg["model"].get("attn_implementation", "eager"),
        device_map=None,
    )

    token_ids = get_structural_token_ids(processor, cfg)
    freeze_backbone(model)
    replaced = replace_qkv_with_script_lora(model, cfg)
    hook_handle = attach_structural_embedding_bank(model, processor, token_ids, cfg)
    model.config.use_cache = False

    if cfg["model"].get("gradient_checkpointing", False):
        enabled = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                enabled = True
            except Exception as exc:  # noqa: BLE001
                print(f"[SAMA] full-model gradient checkpointing failed ({exc!r}); "
                      "falling back to the language model only")
        if not enabled:
            language_model = getattr(model, "language_model", None) or getattr(model, "model", None)
            language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("[SAMA] gradient checkpointing: on")

    trainable, total = count_trainable_parameters(model)
    print(f"[SAMA] replaced QKV modules: {len(replaced)}")
    print(f"[SAMA] trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    if trainable == 0:
        raise RuntimeError("No trainable SAMA parameters found.")

    if cli.init_trainable_checkpoint:
        from sama.modeling import load_trainable_state_dict
        load_trainable_state_dict(model, cli.init_trainable_checkpoint)

    # Keep memory for the attention maps of the final layer only; content of
    # `attentions[-1]` is unchanged, so the structural loss is bit-identical.
    silenced = enable_last_layer_attention(model)
    print(f"[SAMA] attention capture: last layer only ({silenced} layers silenced)")

    template = get_template(
        processor,
        max_pixels=int(cfg["model"]["max_pixels"]),
        max_length=int(cfg["model"].get("max_length", 4096)),
        template_type="qwen2_5_vl",
        # "raise" (the ms-swift default) aborts training on an over-long sample;
        # "right" truncates the tail, keeping the 3 structural tokens and <image>.
        truncation_strategy="right",
    )
    template.set_mode("train")
    # Qwen2.5-VL Template uses the model to calculate MRoPE position IDs in its collator.
    template.model = model

    # Self-check: prove the yaml values actually reached the template.
    print(f"[SAMA] effective max_length={template.max_length} "
          f"max_pixels={template.max_pixels}")

    dataset = SAMADataset(cli.train_jsonl, cfg)
    collator = SAMAMultiViewCollator(template, cfg)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        optim="adamw_torch",
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        save_strategy="steps",
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=bool(train_cfg.get("ddp_find_unused_parameters", True)),
        bf16=bool(train_cfg.get("bf16", True)),
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
        report_to="none",
        logging_first_step=True,
        save_safetensors=False,
        predict_with_generate=False,
        seed=int(train_cfg.get("seed", 42)),
    )

    trainer = SAMASeq2SeqTrainer(
        model=model,
        args=args,
        template=template,
        train_dataset=dataset,
        data_collator=collator,
        sama_cfg=cfg,
        processor=processor,
        token_ids=token_ids,
    )
    trainer.train()
    trainer.save_model(output_dir)
    if hook_handle is not None:
        hook_handle.remove()
    print(f"[SAMA] training finished; artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
