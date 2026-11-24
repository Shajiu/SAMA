from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import regex
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .config import structural_token_name


class SAMADataset(Dataset):
    """JSONL dataset. Each row must contain the SAMA metadata and target answer."""

    def __init__(self, path: str, cfg: dict):
        self.path = str(path)
        self.cfg = cfg
        self.rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                self._validate(row, line_no)
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No data rows found in {path}")

    def _validate(self, row: dict, line_no: int) -> None:
        required = ["image", "answer", "language", "adapter_group", "is_low_resource",
                    "writing_direction", "script_type", "reading_flow"]
        missing = [k for k in required if k not in row]
        if "instruction" not in row and "user_text" not in row:
            missing.append("instruction or user_text")
        if missing:
            raise ValueError(f"{self.path}:{line_no}: missing fields: {missing}")
        if row.get("is_unsafe", False) and not row.get("safety_response") and row.get("task_type") != "jailbreak":
            raise ValueError(
                f"{self.path}:{line_no}: unsafe sample needs safety_response, "
                "or task_type='jailbreak' with answer already being the refusal target."
            )
        group = str(row["adapter_group"])
        allowed = set(self.cfg["lora"]["adapter_groups"]) | {"base"}
        if group not in allowed:
            raise ValueError(f"{self.path}:{line_no}: adapter_group={group!r} not in {sorted(allowed)}")
        for attr, field in [("direction", "writing_direction"), ("script", "script_type"), ("order", "reading_flow")]:
            if row[field] not in self.cfg["structural_vocab"][attr]:
                raise ValueError(
                    f"{self.path}:{line_no}: {field}={row[field]!r} not in "
                    f"{self.cfg['structural_vocab'][attr]}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def _structural_prefix(row: dict, cfg: dict) -> str:
    values = {
        "direction": row["writing_direction"],
        "script": row["script_type"],
        "order": row["reading_flow"],
    }
    tokens = [structural_token_name(attr, values[attr]) for attr in ("direction", "script", "order")]
    return " ".join(tokens)


def _user_body(row: dict, view: str, cfg: dict) -> str:
    if row.get("user_text"):
        body = row["user_text"]
    else:
        instruction_key = "aug_instruction" if view == "aug" and row.get("aug_instruction") else "instruction"
        question_key = "aug_question" if view == "aug" and row.get("aug_question") else "question"
        ocr_key = "aug_ocr_text" if view == "aug" and row.get("aug_ocr_text") else "ocr_text"
        pieces = [row.get(instruction_key, ""), row.get(question_key, ""), row.get(ocr_key, "")]
        body = "\n".join(str(x).strip() for x in pieces if x and str(x).strip())
    if view == "noisy":
        p = float(cfg["loss"].get("noise_probability", 0.15))
        if random.random() < p:
            body = corrupt_text(body, row.get("script_type", "other"))
    return body.strip()


def corrupt_text(text: str, script_type: str = "other") -> str:
    """Conservative grapheme-level OCR corruption; applied to the user-side text only.

    The paper specifies character-level perturbations at p=0.15, but does not enumerate
    exact edit operators. This implementation exposes deletion/duplication/splitting and
    whitespace merging as reproducible engineering choices.
    """
    clusters = regex.findall(r"\X", text)
    if len(clusters) < 2:
        return text
    op = random.choice(["delete", "duplicate", "split", "merge"])
    if op == "delete":
        del clusters[random.randrange(len(clusters))]
    elif op == "duplicate":
        i = random.randrange(len(clusters))
        clusters.insert(i, clusters[i])
    elif op == "split":
        i = random.randrange(1, len(clusters))
        clusters.insert(i, "\u200b")
    elif op == "merge":
        # Remove one existing inter-token whitespace boundary; if none exists, use a
        # zero-width joiner as a conservative segmentation ambiguity marker.
        whitespace = [i for i, c in enumerate(clusters) if c.isspace()]
        if whitespace:
            del clusters[random.choice(whitespace)]
        else:
            i = random.randrange(1, len(clusters))
            clusters.insert(i, "\u200d")
    return "".join(clusters)


def _load_image_for_view(row: dict, view: str, cfg: dict):
    image_path = row.get("aug_image") if view == "aug" and row.get("aug_image") else row["image"]
    if view != "aug" or row.get("aug_image"):
        return image_path
    path = Path(str(image_path))
    if not path.exists():
        # URL inputs are passed through unchanged; provide aug_image for a precomputed
        # layout-preserving variant if augmentation is needed for remote assets.
        return image_path
    image = Image.open(path).convert("RGB")
    aug_cfg = cfg.get("augmentation", {})
    image = ImageEnhance.Brightness(image).enhance(random.uniform(
        float(aug_cfg.get("brightness_min", 0.9)), float(aug_cfg.get("brightness_max", 1.1))))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(
        float(aug_cfg.get("contrast_min", 0.9)), float(aug_cfg.get("contrast_max", 1.1))))
    image = ImageEnhance.Color(image).enhance(random.uniform(
        float(aug_cfg.get("saturation_min", 0.95)), float(aug_cfg.get("saturation_max", 1.05))))
    return image


def make_swift_example(row: dict, view: str, cfg: dict, safety: bool = False) -> dict:
    prefix = _structural_prefix(row, cfg)
    body = _user_body(row, view, cfg)
    user_content = f"{prefix}\n<image>\n{body}"
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": row["system"]})
    messages.append({"role": "user", "content": user_content})

    if safety:
        answer = row.get("safety_response")
        if not answer and row.get("task_type") == "jailbreak":
            answer = row["answer"]
        if not answer:
            raise ValueError("Unsafe row is missing safety_response.")
    else:
        answer = row["answer"]
    messages.append({"role": "assistant", "content": answer})

    image = _load_image_for_view(row, view, cfg)
    return {"messages": messages, "images": [image]}
