#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS=4

# [修改] 指定 Base 模型路径 (原始 Qwen)
export MODEL_LOCAL_PATH="/home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-2B-Instruct"

# [修改] 指定 SFT 微调之后的 LoRA 权重路径
export MODEL_FINETUNE_PATH="./checkpoints/Relevant_charades_frame256_lora3232_bsz3_LR2e4_epoch2_RL_SFT1"

# 数据集配置
export TRAIN_DATA_PATH="/home/fwj/workspace/code/UniTime/UniTime_data/charades/train.json"

export VIDEO_FOLDER="/home/fwj/workspace/VisualSearch/charades/Charades_v1"

# 输出路径
export OUTPUT_DIR="./output/qwen2vl_rl_run1"

export NF_SHORT=128

export ACCELERATE_USE_FSDP=1
export FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="Qwen2VLDecoderLayer"

# 生成默认配置 (防止之前的报错)
if [ ! -f accelerate_config.yaml ]; then
    accelerate config default
fi

accelerate launch --num_processes ${NUM_GPUS} \
    --config_file accelerate_config.yaml \
    train_rl.py