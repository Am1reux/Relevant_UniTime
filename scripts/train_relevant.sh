export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS=4
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

export DECORD_EOF_RETRY_MAX=20480

# MODEL_ID=qwen2-vl-7b-instruct
MODEL_ID=qwen2-vl-2b-instruct
# model_local_path=/home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-7B-Instruct
model_local_path=/home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-2B-Instruct
# TRAIN_DATA_PATH=/home/fwj/workspace/code/UniTime/UniTime_data/charades/train.json
# TRAIN_DATA_PATH=/home/fwj/workspace/code/UniTime/UniTime_data/charades/train.json
TRAIN_DATA_PATH=/home/fwj/workspace/code/UniTime/UniTime_data/qvhl/train.json
# TRAIN_DATA_PATH=./train_debug.json

# EVAL_DATA_PATH=/home/fwj/workspace/code/UniTime/UniTime_data/charades/test.json
# EVAL_DATA_PATH=/home/fwj/workspace/code/UniTime/UniTime_data/charades/val2.0.json
EVAL_DATA_PATH=/home/fwj/workspace/code/UniTime/UniTime_data/qvhl/test2.0.json
IMAGE_FOLDER=None

# VIDEO_FOLDER=/home/fwj/workspace/VisualSearch/charades/Charades_v1 #If you specified video_path in the data file, this can be set to none
VIDEO_FOLDER=/home/fwj/workspace/VisualSearch/qvhighlight/videos #If you specified video_path in the data file, this can be set to none

# FEAT_FOLDER=/home/fwj/workspace/VisualSearch/unitime_feat/charades_7b #If you specified feature_path in the data file, this can be set to none
FEAT_FOLDER=None #If you specified feature_path in the data file, this can be set to none


FPS=2
CLIP_LENGTH=32

TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=False                            # whether train the vision projector (only full finetuning is supported)

USE_LORA=True                                           # whether use lora for llm
Q_LORA=False                                            # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=32                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=32                                            # the lora alpha (both llm and vision encoder)

# RUN_ID=Relevant_Charades_frame1024_lora88_bsz2_LR2e4_epoch2_251218
# RUN_ID=Relevant_charades_frame256_lora3232_bsz3_singleqa_LR2e4_epoch2
# RUN_ID=Relevant_charades_frame256_lora3232_bsz3_LR2e4_epoch2_RL_SFT1
RUN_ID=Relevant_qvhl_frame256_lora3232_bsz3_LR2e4_epoch2_RL_SFT1

DS_STAGE=zero2                                          # deepspeed stage; < zero2 | zero3 >
PER_DEVICE_BATCH_SIZE=1                                # batch size per GPU
GRAD_ACCUM=1                                            # gradient accumulation steps
NUM_EPOCHS=2 # number of training epochs, 1 for tacos, ego4d and pretrain, 2 for charades, anet, qvhl

LR=2e-4                                                 # learning rate
MODEL_MAX_LEN=32768                                      # maximum input length of the model

torchrun $DISTRIBUTED_ARGS train_relevant.py \
    --model_id $MODEL_ID \
    --model_local_path $model_local_path \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --fps $FPS \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to tensorboard \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --save_strategy "epoch" \
    --clip_length $CLIP_LENGTH \
    --feat_folder $FEAT_FOLDER \