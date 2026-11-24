from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def structural_token_name(attribute: str, value: str) -> str:
    return f"<|sama_{attribute}_{value}|>"


def build_structural_token_list(cfg: dict[str, Any]) -> list[str]:
    vocab = cfg["structural_vocab"]
    tokens: list[str] = []
    for attr in ("direction", "script", "order"):
        for value in vocab[attr]:
            tokens.append(structural_token_name(attr, value))
    return tokens
