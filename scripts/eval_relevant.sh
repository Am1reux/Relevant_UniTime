export CUDA_VISIBLE_DEVICES=0,1,2,3
export DECORD_EOF_RETRY_MAX=20480

python inference_relevant.py --model_local_path /home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-2B-Instruct \
    --model_finetune_path ./checkpoints/Relevant_charades_frame256_lora3232_bsz3_LR2e4_epoch2_RL_SFT1 \
    --video_root /home/fwj/workspace/VisualSearch/charades/Charades_v1 \
    --feat_folder None \
    --data_path /home/fwj/workspace/code/UniTime/UniTime_data/charades/test2.0.json \
    --output_dir ./results/test \
    --nf_short 128