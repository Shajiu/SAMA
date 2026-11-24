"""Baseline evaluation: plain Qwen2.5-VL-7B, no SAMA tokens, no LoRA, no structural bank.

Same rows, same user-side text, same decoding settings as tools/eval_sama.py, so the
two logs are directly comparable. Routing accuracy is not meaningful here and is
reported as "-".
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
from pathlib import Path

import torch
from swift import get_model_processor, get_template


def normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def char_f1(prediction: str, gold: str) -> float:
    predicted, expected = normalize(prediction), normalize(gold)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    matcher = difflib.SequenceMatcher(None, list(predicted), list(expected), autojunk=False)
    lcs = sum(block.size for block in matcher.get_matching_blocks())
    precision, recall = lcs / len(predicted), lcs / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_body(row: dict) -> str:
    pieces = [row.get("instruction", ""), row.get("question", ""), row.get("ocr_text", "")]
    return "\n".join(str(piece).strip() for piece in pieces if piece and str(piece).strip())


def resolve_image(path: str, root: str) -> str:
    path = str(path)
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return str(Path(root) / path)


class BaselineInference:
    def __init__(self, checkpoint_dir: str, max_new_tokens_fallback: int = 768):
        artifact = Path(checkpoint_dir) / "sama_config.json"
        with open(artifact, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.max_pixels = int(config["max_pixels"])
        self.max_new_tokens_fallback = max_new_tokens_fallback

        model, processor = get_model_processor(
            config["base_model"],
            model_type="qwen2_5_vl",
            new_special_tokens=[],
            torch_dtype=torch.bfloat16,
            attn_impl="eager",
            device_map=None,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        self.model = model
        self.processor = processor
        self.template = get_template(
            processor,
            max_pixels=self.max_pixels,
            max_length=4096,
            template_type="qwen2_5_vl",
        )
        self.template.set_mode("transformers")
        self.template.model = model

    @torch.inference_mode()
    def generate(self, image_path: str, instruction: str, max_new_tokens: int | None = None) -> str:
        # No structural prefix: this is the untouched model with its native prompt format.
        encoded = self.template.encode({
            "messages": [{"role": "user", "content": f"<image>\n{instruction}"}],
            "images": [image_path],
        })
        batch = self.template.data_collator([encoded])
        batch = {
            key: value for key, value in batch.items()
            if key not in {"labels", "length", "lengths", "loss_scale", "position_ids"}
        }
        device = next(self.model.parameters()).device
        batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}
        output_ids = self.model.generate(
            **batch,
            max_new_tokens=int(max_new_tokens or self.max_new_tokens_fallback),
            do_sample=False,
        )
        new_ids = output_ids[:, batch["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="trained output dir; supplies base_model path")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    engine = BaselineInference(args.checkpoint)
    rows: list[dict] = []
    with open(args.jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No rows in {args.jsonl}")
    if args.limit:
        rows = rows[: args.limit]

    scores: dict[str, list[float]] = collections.defaultdict(list)
    exact_hits = 0
    details: list[dict] = []

    for index, row in enumerate(rows, start=1):
        image = resolve_image(row["image"], args.root)
        prediction = engine.generate(image, build_body(row), args.max_new_tokens)
        score = char_f1(prediction, row["answer"])
        exact = float(normalize(prediction) == normalize(row["answer"]))
        exact_hits += exact
        scores["all"].append(score)
        scores[str(row.get("adapter_group", "?"))].append(score)
        details.append({
            "id": row.get("id"),
            "image": row["image"],
            "adapter_group": row.get("adapter_group"),
            "answer": row["answer"],
            "prediction": prediction,
            "char_f1": score,
            "exact": exact,
        })
        if index % 20 == 0 or index == len(rows):
            print(f"  ... {index}/{len(rows)}  running char_f1={sum(scores['all'])/len(scores['all']):.4f}", flush=True)

    print("\nper adapter_group:")
    for group in sorted(scores):
        if group == "all":
            continue
        values = scores[group]
        print(f"  {group:10s} n={len(values):5d}  char_f1={sum(values)/len(values):.4f}")
    total = len(rows)
    print(f"\noverall  n={total}  char_f1={sum(scores['all'])/total:.4f}  EM={exact_hits/total:.4f}  route_acc=-")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for item in details:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"per-row predictions -> {args.out}")


if __name__ == "__main__":
    main()
