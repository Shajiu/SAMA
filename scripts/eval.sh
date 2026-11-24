#!/usr/bin/env bash
# =============================================================================
# SAMA 批量评测脚本（字符级 F1 / EM / 路由准确率，按 adapter_group 分桶）
#
# 用法：
#   bash scripts/eval.sh data/sama/val.jsonl --stratified 40
#   JSONL=data/sama_unsafe/test.jsonl MAX_NEW=128 bash scripts/eval.sh "$JSONL"
#
# 评测集与训练集必须按图片划分，不能有重叠。
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
JSONL="${JSONL:-${1:-data/sama/val.jsonl}}"
MAX_NEW="${MAX_NEW:-768}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
shift || true

# -u 关闭输出缓冲：否则用管道看日志时会一片空白，直到进程结束
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -u tools/eval_sama.py \
  --checkpoint "$CKPT" \
  --jsonl "$JSONL" \
  --root . \
  --max_new_tokens "$MAX_NEW" \
  "$@"
