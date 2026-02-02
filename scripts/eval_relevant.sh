export CUDA_VISIBLE_DEVICES=0,1,2
export DECORD_EOF_RETRY_MAX=20480

python inference_relevant.py --model_local_path /home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-2B-Instruct \
    --model_finetune_path ./checkpoints/Relevant_charades_frame256_lora3232_bsz3_singleqa_LR2e4_epoch2 \
    --video_root /home/fwj/workspace/VisualSearch/charades/Charades_v1 \
    --feat_folder None \
    --data_path /home/fwj/workspace/code/UniTime/UniTime_data/charades/test2.0.json \
    --output_dir ./results/Relevant_charades_frame256_lora3232_bsz3_singleqa_LR2e4_epoch2 \
    --nf_short 128