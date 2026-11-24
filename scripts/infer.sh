#!/usr/bin/env bash
# =============================================================================
# SAMA 单图推理脚本
#
# 用法：
#   bash scripts/infer.sh data/sama/images/.../xxx.png "请阅读图像中的文字并回答问题。"
#   CKPT=output/sama_7b_full_v3 CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh xxx.png
#
# 可选环境变量：
#   CKPT        训练产物目录（必须含 sama_trainable.pt / sama_config.json / processor/）
#   CONFIG      配置文件，默认 configs/sama_7b.yaml
#   HINT        语种提示 ug|bo|mn|zh|en，用于选择 script / 结构 token / LoRA 分支
#   MAX_NEW     最大生成 token，默认 512
#
# 输出是一段 JSON：response（模型回答）+ script + writing_direction + reading_flow
# + adapter_group（实际激活的 LoRA 分支）。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export no_proxy=localhost,127.0.0.1,10.0.0.0/8
export http_proxy=http://127.0.0.1:9
export https_proxy=http://127.0.0.1:9
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

CKPT="${CKPT:-output/sama_7b_full}"
CONFIG="${CONFIG:-configs/sama_7b.yaml}"
MAX_NEW="${MAX_NEW:-512}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

IMAGE="${1:?用法: bash scripts/infer.sh <图片路径> [指令]}"
INSTRUCTION="${2:-请阅读图像中的文字，并回答下面的问题。}"

ARGS=(--config "$CONFIG" --checkpoint "$CKPT" --image "$IMAGE" --instruction "$INSTRUCTION" --max_new_tokens "$MAX_NEW")
if [ -n "${HINT:-}" ]; then
  ARGS+=(--language_hint "$HINT")
fi
if [ -n "${OCR_TEXT:-}" ]; then
  ARGS+=(--ocr_text "$OCR_TEXT")
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python infer_sama.py "${ARGS[@]}"
