#!/usr/bin/env bash
# =============================================================================
# SAMA 训练启动脚本
#
# 用法：
#   bash scripts/train.sh                     # 用下面的默认值
#   GPUS=0,1,2,3 NUM=4 bash scripts/train.sh  # 改卡号/进程数
#   TRAIN_JSONL=data/sama/train_small.jsonl OUTPUT=output/try1 bash scripts/train.sh
#
# 必须在仓库根目录执行（脚本会自动 cd 到上级目录），因为 JSONL 里的图片是
# 相对仓库根目录的路径。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
# 三视图同时存活 + 反复分配，碎片化严重，必须开可扩展段
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 离线机器：禁止任何联网检查（否则 urllib3 会挂到系统超时，看起来像"卡死"）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export no_proxy=localhost,127.0.0.1,10.0.0.0/8
# 把代理指向一个必然拒绝的本地端口：连接立刻失败，而不是等 130 秒再重试 4 次
export http_proxy=http://127.0.0.1:9
export https_proxy=http://127.0.0.1:9
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

GPUS="${GPUS:-4,5,6,7}"          # 物理卡号，逗号分隔
NUM="${NUM:-4}"                  # 进程数，必须等于上面的卡数
CONFIG="${CONFIG:-configs/sama_7b.yaml}"
TRAIN_JSONL="${TRAIN_JSONL:-data/sama/train_full.jsonl}"
OUTPUT="${OUTPUT:-output/sama_7b_full}"

mkdir -p "$(dirname "$OUTPUT")"
LOG="sama_train_$(date +%Y%m%d_%H%M%S).log"

echo "GPUS=$GPUS NUM=$NUM CONFIG=$CONFIG"
echo "TRAIN_JSONL=$TRAIN_JSONL OUTPUT=$OUTPUT LOG=$LOG"

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
  --num_processes "$NUM" \
  --mixed_precision bf16 \
  train_sama.py \
  --config "$CONFIG" \
  --train_jsonl "$TRAIN_JSONL" \
  --output_dir "$OUTPUT" \
  2>&1 | tee "$LOG"

# 续训：加 --init_trainable_checkpoint output/xxx/sama_trainable.pt
# 起跑后应看到这四行，缺一行就说明配置没生效：
#   [SAMA] gradient checkpointing: on
#   [SAMA] replaced QKV modules: 84
#   [SAMA] trainable parameters: 82,632,704 / 8,374,799,360 (0.9867%)
#   [SAMA] attention capture: last layer only (27 layers silenced)
#   [SAMA] effective max_length=4096 max_pixels=262144
