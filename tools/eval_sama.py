"""Evaluate a trained SAMA checkpoint on a JSONL split: EM, char-F1, routing accuracy.

Reports per adapter_group so the three LoRA branches can be compared against each
other and against a base-model baseline.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
from pathlib import Path

from infer_sama import SAMAInference


def normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def resolve_image(path: str, root: str) -> str:
    """Accept absolute paths, cwd-relative paths, or paths relative to --root."""
    path = str(path)
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return str(Path(root) / path)


def char_f1(prediction: str, gold: str) -> float:
    """LCS-based character F1; whitespace-insensitive and order-sensitive."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sama_7b.yaml")
    parser.add_argument("--checkpoint", required=True, help="e.g. output/sama_7b_full")
    parser.add_argument("--jsonl", required=True, help="e.g. data/sama/val.jsonl")
    parser.add_argument("--root", default=".", help="base directory for relative image paths")
    parser.add_argument("--limit", type=int, default=0, help="0 = every row")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out", default=None, help="optional JSONL of per-row predictions")
    args = parser.parse_args()

    engine = SAMAInference(args.config, args.checkpoint)
    rows: list[dict] = []
    skipped = 0
    with open(args.jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if not rows:
        raise SystemExit(
            f"No JSON rows found in {args.jsonl}; scoring needs rows with an 'answer' field, not a bare image list."
        )
    if skipped:
        print(f"  skipped {skipped} non-JSON line(s)")
    if args.limit:
        rows = rows[: args.limit]

    scores: dict[str, list[float]] = collections.defaultdict(list)
    exact_hits = 0
    route_hits = 0
    details: list[dict] = []

    for index, row in enumerate(rows, start=1):
        image = resolve_image(row["image"], args.root)
        instruction = row.get("instruction") or row.get("user_text") or ""
        result = engine.generate(
            image,
            instruction,
            language_hint=row.get("language"),
            ocr_text=row.get("ocr_text", ""),
            ocr_regions=row.get("ocr_regions"),
            max_new_tokens=args.max_new_tokens,
        )
        prediction = result.get("response", "")
        score = char_f1(prediction, row["answer"])
        exact = float(normalize(prediction) == normalize(row["answer"]))
        routed = float(result.get("adapter_group") == row.get("adapter_group"))
        exact_hits += exact
        route_hits += routed
        scores["all"].append(score)
        scores[str(row.get("adapter_group", "?"))].append(score)
        details.append({
            "id": row.get("id"),
            "image": row["image"],
            "adapter_group": row.get("adapter_group"),
            "script_type": row.get("script_type"),
            "predicted_group": result.get("adapter_group"),
            "predicted_script": result.get("script"),
            "answer": row["answer"],
            "prediction": prediction,
            "char_f1": score,
            "exact": exact,
            "routed": routed,
        })
        if index % 20 == 0 or index == len(rows):
            running = sum(scores["all"]) / len(scores["all"])
            print(f"  ... {index}/{len(rows)}  running char_f1={running:.4f}", flush=True)

    print("\nper adapter_group:")
    for group in sorted(scores):
        if group == "all":
            continue
        values = scores[group]
        print(f"  {group:10s} n={len(values):5d}  char_f1={sum(values)/len(values):.4f}")
    total = len(rows)
    print(
        f"\noverall  n={total}  char_f1={sum(scores['all'])/total:.4f}"
        f"  EM={exact_hits/total:.4f}  route_acc={route_hits/total:.4f}"
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for item in details:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"per-row predictions -> {args.out}")


if __name__ == "__main__":
    main()
