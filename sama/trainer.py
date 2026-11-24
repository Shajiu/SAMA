"""SAMA trainer: three-view losses, adapter routing, and DDP-safe step structure.

Why this file is custom
-----------------------
1. A batch may need up to four forward passes (clean / augmented / noisy / safety).
   Under DDP every rank must issue the *same* sequence of collectives, so the number
   of forwards per optimisation step has to be identical across ranks. We therefore
   AND-reduce the data-dependent flags (`is_low_resource`, `has_safety`) with a
   single all_reduce, and substitute the clean view where a rank has no data for a
   view. This keeps the structure identical without touching the loss values.
2. DDP's reducer setup (`prepare_for_backward`) must not run once per forward. All
   views except the last are executed under `model.no_sync()`; only the final
   forward is DDP-synced, so one backward triggers exactly one gradient all-reduce.
3. ms-swift's mixin passes `data_collator` both positionally and by keyword, which
   `transformers` rejects -- we hand it over exactly once.
4. Only the trainable SAMA tensors are saved (`sama_trainable.pt`), never the full
   backbone.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from swift.trainers import Seq2SeqTrainer

from .losses import structure_js_loss, symmetric_kl_from_logits, total_sama_loss, zero_loss
from .modeling import save_sama_artifacts, set_active_adapter
from .sampler import GroupedBatchSampler


class _OrderedBatchList(list):
    """A materialised batch order that still advertises `batch_size` to accelerate.

    `accelerate` requires a batch sampler exposing `.batch_size`; a plain list makes
    `BatchSamplerShard` raise "You need to use even_batches=False when the batch
    sampler has no batch size".
    """

    def __init__(self, batches, batch_size):
        super().__init__(batches)
        self.batch_size = batch_size


class SAMASeq2SeqTrainer(Seq2SeqTrainer):
    """ms-swift Seq2SeqTrainer extended for SAMA's paired multimodal objectives."""

    def __init__(self, *args, sama_cfg: dict, processor, token_ids: dict, **kwargs):
        data_collator = kwargs.pop("data_collator", None)
        super().__init__(*args, **kwargs)
        if data_collator is not None:
            self.data_collator = data_collator
        self.sama_cfg = sama_cfg
        self.processor = processor
        self.token_ids = token_ids
        image_token_id = getattr(self.model.config, "image_token_id", None)
        self.image_token_id = int(image_token_id if image_token_id is not None else 151655)
        self.last_sama_losses: dict[str, float] = {}
        self._aux_sum: dict[str, float] = {}
        self._aux_n = 0

    # ------------------------------------------------------------------ data
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("SAMA training requires train_dataset.")
        batch_sampler = GroupedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            seed=self.args.seed,
            drop_last=self.args.dataloader_drop_last,
        )
        ordered = _OrderedBatchList(
            self._homogeneous_step_order(batch_sampler),
            getattr(batch_sampler, "batch_size", None),
        )
        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=ordered,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return self.accelerator.prepare(dataloader)

    def _homogeneous_step_order(self, batch_sampler) -> list[list[int]]:
        """Interleave batches so that one DDP step sees one batch class on every rank.

        accelerate shards batches round-robin (`batch i -> rank i % replicas`), so a
        step consumes `replicas` consecutive batches. Emitting them in chunks of
        `replicas` from one `(is_low_resource, is_unsafe)` bucket guarantees that all
        ranks run the same view structure in that step -- without this, one rank on a
        base-only batch forces every rank to skip the structural/robust views.
        """
        replicas = max(1, int(self.accelerator.num_processes))
        buckets: dict[tuple[bool, bool], list[list[int]]] = {}
        for batch in batch_sampler:
            row = self.train_dataset[int(batch[0])]
            key = (bool(row["is_low_resource"]), bool(row.get("is_unsafe", False)))
            buckets.setdefault(key, []).append(list(batch))
        ordered: list[list[int]] = []
        cursor = {key: 0 for key in buckets}
        while any(cursor[key] < len(buckets[key]) for key in buckets):
            for key in buckets:
                chunk = buckets[key][cursor[key]:cursor[key] + replicas]
                if chunk:
                    ordered.extend(chunk)
                    cursor[key] += len(chunk)
        return ordered

    # ------------------------------------------------------------------ loss
    @staticmethod
    def _without_labels(batch: dict) -> dict:
        return {k: v for k, v in batch.items() if k != "labels"}

    @staticmethod
    def _sync_step_flags(flags: list[bool], device) -> list[bool]:
        """AND-reduce flags across ranks so every rank runs the same forward structure."""
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return flags
        tensor = torch.tensor([1 if f else 0 for f in flags], device=device, dtype=torch.uint8)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return [bool(v) for v in tensor.tolist()]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        meta = inputs["meta"]
        route_group = meta["adapter_group"]
        active = None if route_group == "base" else route_group
        unwrapped = self.accelerator.unwrap_model(model)
        set_active_adapter(unwrapped, active)

        # SAMA only reads the outputs (logits are sliced, never modified in place), so
        # DDP's defensive clone of every output tensor can be skipped -- this removes a
        # ~1 GiB peak and a few percent of step time.
        if getattr(model, "_ddp_sink_clone", None) is not False:
            setter = getattr(model, "_set_ddp_sink_clone", None)
            if setter is not None:
                setter(False)
            elif hasattr(model, "_ddp_sink_clone"):
                model._ddp_sink_clone = False

        clean_batch = inputs["clean"]
        device = clean_batch["input_ids"].device
        is_low_resource, has_safety = self._sync_step_flags(
            [bool(meta["is_low_resource"]), inputs.get("safety") is not None], device
        )

        # A rank that was not scheduled for a view still runs it on the clean batch:
        # identical views give a structural loss of exactly zero, and the view count
        # stays uniform across ranks.
        aug_batch = inputs.get("aug") or clean_batch
        noisy_batch = inputs.get("noisy") or clean_batch
        safety_batch = inputs.get("safety") or clean_batch

        # (forward kwargs, output_attentions) -- only the last one may be DDP-synced.
        views: list[tuple[dict, bool]] = [(clean_batch, is_low_resource)]
        if is_low_resource:
            views.append((self._without_labels(aug_batch), True))
            views.append((self._without_labels(noisy_batch), False))
        if has_safety:
            views.append((safety_batch, False))

        no_sync = getattr(model, "no_sync", None)
        outputs = []
        for index, (batch, want_attn) in enumerate(views):
            last = index == len(views) - 1
            ctx = nullcontext() if (last or no_sync is None) else no_sync()
            with ctx, self.template.forward_context(model, batch):
                outputs.append(model(**batch, output_attentions=want_attn, use_cache=False))

        clean_outputs = outputs[0]
        loss_task = clean_outputs.loss
        if loss_task is None:
            raise RuntimeError(
                "The clean SFT forward did not return loss; check `labels` from the Swift template."
            )
        loss_struct = zero_loss(clean_outputs.logits)
        loss_robust = zero_loss(clean_outputs.logits)
        loss_safe = zero_loss(clean_outputs.logits)

        cursor = 1
        if is_low_resource:
            loss_struct = structure_js_loss(
                clean_outputs, outputs[cursor], clean_batch, aug_batch, self.image_token_id
            )
            cursor += 1
            loss_robust = symmetric_kl_from_logits(
                clean_outputs, outputs[cursor], clean_batch, noisy_batch
            )
            cursor += 1
        if has_safety:
            safety_outputs = outputs[cursor]
            if safety_outputs.loss is not None:
                loss_safe = safety_outputs.loss

        loss = total_sama_loss(loss_task, loss_struct, loss_safe, loss_robust, self.sama_cfg)

        values = {
            "loss_task": loss_task,
            "loss_struct": loss_struct,
            "loss_robust": loss_robust,
            "loss_safe": loss_safe,
            "loss_total": loss,
        }
        for key, value in values.items():
            self._aux_sum[key] = self._aux_sum.get(key, 0.0) + float(value.detach().item())
        self._aux_n += 1
        self.last_sama_losses = {k: float(v.detach().item()) for k, v in values.items()}
        return (loss, clean_outputs) if return_outputs else loss

    # ------------------------------------------------------------------ logging
    def log(self, logs: dict, *args, **kwargs):
        # One optimisation step spans several micro-batches; report the mean instead of
        # whichever micro-batch happened to be last.
        if self._aux_n:
            for key, total in self._aux_sum.items():
                mean = total / self._aux_n
                logs.setdefault(key, mean)
                self.last_sama_losses[key] = mean
            self._aux_sum = {}
            self._aux_n = 0
        for key, value in self.last_sama_losses.items():
            if key not in logs:
                scalar = value.item() if hasattr(value, "item") else float(value)
                logs[key] = float(scalar)
        return super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------ artifacts
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        if not self.args.should_save:
            return
        unwrapped = self.accelerator.unwrap_model(self.model)
        save_sama_artifacts(unwrapped, self.processor, self.sama_cfg, self.token_ids, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        with open(Path(output_dir) / "trainer_args.json", "w", encoding="utf-8") as handle:
            json.dump(self.args.to_dict(), handle, ensure_ascii=False, indent=2, default=str)
