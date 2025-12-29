export CUDA_VISIBLE_DEVICES=2,3
export DECORD_EOF_RETRY_MAX=20480

python inference_relevant.py --model_local_path /home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-7B-Instruct \
    --model_finetune_path /home/fwj/workspace/code/UniTime/checkpoints/Relevant_Charades_frame1024_lora88_bsz2_LR2e4_epoch2_rematch_1 \
    --video_root /home/fwj/workspace/VisualSearch/charades/Charades_v1 \
    --feat_folder /home/fwj/workspace/VisualSearch/unitime_feat/charades_7b \
    --data_path /home/fwj/workspace/code/UniTime/UniTime_data/charades/test2.0.json \
    --output_dir /home/fwj/workspace/code/UniTime/results/Relevant_charades_frame1024_lora88_bsz2_LR2e4_epoch2_rematch_1 \
    --nf_short 128