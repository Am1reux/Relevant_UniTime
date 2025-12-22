export CUDA_VISIBLE_DEVICES=1
python inference.py --model_local_path /home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-7B-Instruct \
     --model_finetune_path /home/fwj/workspace/code/UniTime/UniTime_ckpt \
     --data_path data/test.json \
     --output_dir ./results/test \
     --nf_short 128