#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

# 必须指向你 SFT 后的模型路径，而不是原始模型！
# RL 是在 SFT 模型基础上的二阶段优化。
export MODEL_PATH="/home/fwj/workspace/Relevant_UniTime/output/your_sft_checkpoint" 

export TRAIN_DATA_PATH="/home/fwj/workspace/code/UniTime/UniTime_data/charades/train.json"
export VIDEO_FOLDER="/home/fwj/workspace/VisualSearch/charades/Charades_v1"
export OUTPUT_DIR="./output/qwen2vl_rl_run1"

# 解决多进程 context 问题
export ACCELERATE_USE_FSDP=1
export FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="Qwen2VLDecoderLayer"

# 启动训练
# GRPO 比较吃显存，因为需要 inference G 次，建议 num_processes=3 利用你的3张卡
accelerate launch --num_processes 3 \
    --config_file accelerate_config.yaml \ # 如果有的话，或者去掉这行让它自动配置
    train_rl.py