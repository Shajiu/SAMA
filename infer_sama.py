from __future__ import annotations

import argparse
import json
from pathlib import Path

import regex
import torch
from swift import get_model_processor, get_template

from sama.config import build_structural_token_list, load_config, structural_token_name
from sama.modeling import (
    get_structural_token_ids,
    attach_structural_embedding_bank,
    freeze_backbone,
    load_trainable_state_dict,
    replace_qkv_with_script_lora,
    set_active_adapter,
)


SCRIPT_PATTERNS = {
    "arabic": regex.compile(r"\p{Script=Arabic}"),
    "tibetan": regex.compile(r"\p{Script=Tibetan}"),
    "mongolian": regex.compile(r"\p{Script=Mongolian}"),
    "latin": regex.compile(r"\p{Script=Latin}"),
    "han": regex.compile(r"\p{Script=Han}"),
}


def infer_script(text: str, language_hint: str | None = None, ocr_regions: list[dict] | None = None) -> str:
    # When OCR regions are supplied, follow the paper's dominant-region rule:
    # select the largest detected text region, then identify its Unicode script.
    if ocr_regions:
        def region_area(region):
            if region.get("area") is not None:
                return float(region["area"])
            bbox = region.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return abs(float(bbox[2]) - float(bbox[0])) * abs(float(bbox[3]) - float(bbox[1]))
            return float(len(region.get("text", "")))
        largest = max(ocr_regions, key=region_area)
        text = str(largest.get("text", text or ""))
    counts = {name: len(pattern.findall(text or "")) for name, pattern in SCRIPT_PATTERNS.items()}
    best = max(counts, key=counts.get) if counts else "other"
    if counts.get(best, 0) > 0:
        return best
    # Fallback to the paper's language/script signals when OCR text is unavailable.
    hint_map = {"ug": "arabic", "bo": "tibetan", "mn": "mongolian", "zh": "han", "en": "latin"}
    if language_hint in hint_map:
        return hint_map[language_hint]
    return "other"


def structure_for_script(script: str, cfg: dict) -> dict:
    defaults = {
        "arabic": ("rtl", "rtl"),
        "tibetan": ("ltr", "ltr"),
        "mongolian": ("vertical", "top_to_bottom"),
        "latin": ("ltr", "ltr"),
        "han": ("ltr", "ltr"),
        "other": ("ltr", "ltr"),
    }
    direction, order = defaults.get(script, ("mixed", "mixed"))
    adapter = cfg["script_to_adapter"].get(script, "base")
    if adapter is None:
        raise ValueError(f"No adapter mapping configured for script={script!r}")
    return {
        "writing_direction": direction,
        "script_type": script if script in cfg["structural_vocab"]["script"] else "other",
        "reading_flow": order,
        "adapter_group": adapter,
    }


class SAMAInference:
    def __init__(self, config_path: str, checkpoint_dir: str):
        self.cfg = load_config(config_path)
        checkpoint_dir = Path(checkpoint_dir)
        artifact_cfg_path = checkpoint_dir / "sama_config.json"
        if not artifact_cfg_path.exists():
            raise FileNotFoundError(f"Missing {artifact_cfg_path}")
        with open(artifact_cfg_path, "r", encoding="utf-8") as f:
            self.artifact_cfg = json.load(f)
        # Route with the mapping saved alongside the trained adapter bank.
        self.cfg["script_to_adapter"] = self.artifact_cfg.get("script_to_adapter", self.cfg["script_to_adapter"])

        model, processor = get_model_processor(
            self.artifact_cfg["base_model"],
            model_type="qwen2_5_vl",
            new_special_tokens=build_structural_token_list(self.cfg),
            torch_dtype=torch.bfloat16,
            attn_impl="eager",
            device_map=None,
        )
        # Use the exact token vocabulary recorded by the training config.
        self.token_ids = get_structural_token_ids(processor, self.cfg)
        saved_token_ids = self.artifact_cfg.get("token_ids", {})
        for token, token_id in saved_token_ids.items():
            if token in self.token_ids and int(token_id) != int(self.token_ids[token]):
                raise RuntimeError(f"Structural token id mismatch for {token}: checkpoint={token_id}, current={self.token_ids[token]}")
        freeze_backbone(model)
        replace_qkv_with_script_lora(model, self.cfg)
        self.hook_handle = attach_structural_embedding_bank(model, processor, self.token_ids, self.cfg)
        load_trainable_state_dict(model, checkpoint_dir / "sama_trainable.pt")
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        self.model = model
        self.processor = processor
        self.template = get_template(
            processor,
            max_pixels=int(self.artifact_cfg["max_pixels"]),
            max_length=int(self.cfg["model"].get("max_length", 4096)),
            template_type="qwen2_5_vl",
        )
        self.template.set_mode("transformers")
        # Required by Qwen2.5-VL Template when building MRoPE position IDs.
        self.template.model = self.model

    @torch.inference_mode()
    def generate(self, image_path: str, instruction: str, language_hint: str | None = None,
                 ocr_text: str = "", ocr_regions: list[dict] | None = None,
                 max_new_tokens: int | None = None) -> dict:
        script = infer_script(ocr_text, language_hint, ocr_regions)
        structure = structure_for_script(script, self.cfg)
        group = structure["adapter_group"]
        set_active_adapter(self.model, None if group == "base" else group)

        prefix = " ".join([
            structural_token_name("direction", structure["writing_direction"]),
            structural_token_name("script", structure["script_type"]),
            structural_token_name("order", structure["reading_flow"]),
        ])
        user_text = f"{prefix}\n<image>\n{instruction}"
        if ocr_text:
            user_text += f"\nOCR text: {ocr_text}"
        encoded = self.template.encode({
            "messages": [{"role": "user", "content": user_text}],
            "images": [image_path],
        })
        batch = self.template.data_collator([encoded])
        # The ms-swift template pre-computes MRoPE `position_ids` for the prompt.
        # `generate` only appends to `input_ids` and never extends that tensor, so
        # the first decoded token would get a missing rope position (shapes 115 vs
        # 114) -- drop it and let the model rebuild positions itself.
        batch = {k: v for k, v in batch.items() if k not in {"labels", "length", "lengths", "loss_scale", "position_ids"}}
        device = next(self.model.parameters()).device
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        output_ids = self.model.generate(
            **batch,
            max_new_tokens=int(max_new_tokens or self.cfg["inference"]["max_new_tokens"]),
            do_sample=bool(self.cfg["inference"].get("do_sample", False)),
            temperature=float(self.cfg["inference"].get("temperature", 0.0)),
        )
        input_len = batch["input_ids"].shape[1]
        new_ids = output_ids[:, input_len:]
        text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        return {"response": text, "script": script, **structure}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sama_7b.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--language_hint", default=None)
    parser.add_argument("--ocr_text", default="")
    parser.add_argument("--ocr_regions_json", default=None,
                        help="Optional JSON file containing OCR regions [{text, bbox/area}, ...].")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    args = parser.parse_args()

    ocr_regions = None
    if args.ocr_regions_json:
        with open(args.ocr_regions_json, "r", encoding="utf-8") as f:
            ocr_regions = json.load(f)
    engine = SAMAInference(args.config, args.checkpoint)
    result = engine.generate(
        args.image, args.instruction, args.language_hint, args.ocr_text,
        ocr_regions=ocr_regions, max_new_tokens=args.max_new_tokens
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
