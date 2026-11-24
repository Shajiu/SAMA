from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from .config import build_structural_token_list, structural_token_name


class ScriptLoRALinear(nn.Module):
    """A frozen Linear layer plus one independently selectable LoRA branch per script group.

    The active adapter is selected once per homogeneous batch by SAMASeq2SeqTrainer.
    This deliberately implements the paper's single-branch-per-input inference behavior.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_groups: Iterable[str],
        rank: int,
        alpha: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hasattr(base_layer, "in_features") or not hasattr(base_layer, "out_features"):
            raise TypeError(f"Expected an nn.Linear-like module, got {type(base_layer)}")
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.active_adapter: str | None = None

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        for group in adapter_groups:
            safe_group = str(group)
            a = nn.Linear(self.in_features, self.rank, bias=False)
            b = nn.Linear(self.rank, self.out_features, bias=False)
            # Keep newly created branch parameters beside the wrapped base projection.
            base_weight = getattr(base_layer, "weight", None)
            if base_weight is not None:
                a = a.to(device=base_weight.device, dtype=base_weight.dtype)
                b = b.to(device=base_weight.device, dtype=base_weight.dtype)
            nn.init.kaiming_uniform_(a.weight, a=5**0.5)
            nn.init.zeros_(b.weight)
            self.lora_A[safe_group] = a
            self.lora_B[safe_group] = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        group = self.active_adapter
        if group is not None:
            delta = self.lora_B[group](self.lora_A[group](self.dropout(x)))
            result = result + delta * self.scaling
        return result


class StructuralEmbeddingBank(nn.Module):
    """Trainable vectors replacing dedicated structural marker-token embeddings."""

    def __init__(self, token_id_to_key: dict[int, str], base_embedding: nn.Embedding):
        super().__init__()
        self.token_id_to_key = {int(k): str(v) for k, v in token_id_to_key.items()}
        self.vectors = nn.ParameterDict()
        for token_id, key in self.token_id_to_key.items():
            init = base_embedding.weight[token_id].detach().clone()
            self.vectors[key] = nn.Parameter(init)

    def replace(self, input_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        # No in-place assignment: keeps the graph for structural vectors intact.
        out = embeddings
        for token_id, key in self.token_id_to_key.items():
            mask = input_ids.eq(token_id).unsqueeze(-1)
            vector = self.vectors[key].to(dtype=out.dtype, device=out.device)
            out = torch.where(mask, vector.view(1, 1, -1), out)
        return out

    def hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return output
        input_ids = inputs[0]
        if input_ids.ndim != 2:
            return output
        return self.replace(input_ids, output)


def _safe_key(attribute: str, value: str) -> str:
    return f"{attribute}__{value}".replace("-", "_")


def get_structural_token_ids(processor, cfg: dict) -> dict[str, int]:
    """Read IDs for tokens registered through Swift's `new_special_tokens` loader option.

    Let ms-swift perform token registration and padded embedding resize: its model loader
    includes Qwen2.5-VL-specific vocabulary-size handling (including 128-row alignment).
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The Swift processor has no .tokenizer; expected Qwen2.5-VLProcessor.")
    tokens = build_structural_token_list(cfg)
    result = {}
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or (unk_id is not None and int(token_id) == int(unk_id)):
            raise RuntimeError(
                f"Structural token {token!r} is missing. Pass the full token list through "
                "get_model_processor(..., new_special_tokens=...)."
            )
        result[token] = int(token_id)
    return result


def freeze_backbone(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def replace_qkv_with_script_lora(model, cfg: dict) -> list[str]:
    lora_cfg = cfg["lora"]
    groups = list(lora_cfg["adapter_groups"])
    targets = set(lora_cfg["target_modules"])

    replacements: list[tuple[str, nn.Module]] = []
    for name, module in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        # Restrict replacement to the LLM self-attention projections, not the vision tower.
        if leaf in targets and ".self_attn." in name and "language_model.layers." in name:
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Target {name} is not nn.Linear: {type(module)}")
            replacements.append((name, module))

    if not replacements:
        candidate_names = [n for n, _ in model.named_modules() if n.endswith(tuple(f".{x}" for x in targets))]
        raise RuntimeError(
            "Could not locate Qwen2.5-VL LLM q_proj/k_proj/v_proj modules. "
            f"Candidate module names: {candidate_names[:20]}"
        )

    for full_name, base_layer in replacements:
        parent = model
        parts = full_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(
            parent,
            parts[-1],
            ScriptLoRALinear(
                base_layer=base_layer,
                adapter_groups=groups,
                rank=int(lora_cfg["rank"]),
                alpha=int(lora_cfg["alpha"]),
                dropout=float(lora_cfg.get("dropout", 0.0)),
            ),
        )
    return [name for name, _ in replacements]


def attach_structural_embedding_bank(model, processor, token_ids: dict[str, int], cfg: dict):
    tokenizer = processor.tokenizer
    embedding = model.get_input_embeddings()
    token_id_to_key: dict[int, str] = {}
    for attr in ("direction", "script", "order"):
        for value in cfg["structural_vocab"][attr]:
            token = structural_token_name(attr, value)
            token_id = token_ids[token]
            token_id_to_key[token_id] = _safe_key(attr, value)
    bank = StructuralEmbeddingBank(token_id_to_key, embedding)
    model.add_module("sama_structural_bank", bank)
    handle = embedding.register_forward_hook(bank.hook)
    return handle


def set_active_adapter(model, adapter_name: str | None) -> None:
    for module in model.modules():
        if isinstance(module, ScriptLoRALinear):
            module.active_adapter = adapter_name


def count_trainable_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(model, checkpoint_path: str | Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "trainable_state" in state:
        state = state["trainable_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Frozen base-model weights are intentionally absent; only trainable tensors must match.
    unexpected_trainable = [k for k in unexpected if "lora_" in k or "sama_structural_bank" in k]
    if unexpected_trainable:
        raise RuntimeError(f"Unexpected trainable checkpoint keys: {unexpected_trainable[:20]}")
    loaded = set(state)
    required = {n for n, p in model.named_parameters() if p.requires_grad}
    absent = sorted(required - loaded)
    if absent:
        raise RuntimeError(f"Checkpoint misses trainable parameters: {absent[:20]}")


def save_sama_artifacts(model, processor, cfg: dict, token_ids: dict[str, int], output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable_state": trainable_state_dict(model)}, output_dir / "sama_trainable.pt")
    processor.save_pretrained(output_dir / "processor")
    artifact_cfg = {
        "base_model": cfg["model"]["model_name_or_path"],
        "adapter_groups": cfg["lora"]["adapter_groups"],
        "rank": cfg["lora"]["rank"],
        "alpha": cfg["lora"]["alpha"],
        "dropout": cfg["lora"].get("dropout", 0.0),
        "target_modules": cfg["lora"]["target_modules"],
        "structural_vocab": cfg["structural_vocab"],
        "token_ids": token_ids,
        "script_to_adapter": cfg["script_to_adapter"],
        "max_pixels": cfg["model"]["max_pixels"],
    }
    with open(output_dir / "sama_config.json", "w", encoding="utf-8") as f:
        json.dump(artifact_cfg, f, ensure_ascii=False, indent=2)
