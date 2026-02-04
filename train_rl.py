import os
import sys
import torch
import transformers
from trl import GRPOTrainer, GRPOConfig
from datasets_mr_relevant import VideoCentricDataset
from loaders import LOADERS
from collators import COLLATORS
from reward_utils import unified_reward_func
from peft import PeftModel # [新增] 引入 PeftModel

def main():
    parser = transformers.HfArgumentParser((GRPOConfig))
    
    # [修改] 区分 Base 模型路径和 SFT 微调路径，与 inference 对齐
    # 原始 Qwen2-VL 基座路径
    model_local_path = os.getenv("MODEL_LOCAL_PATH", "/home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-2B-Instruct")
    # SFT LoRA 微调路径
    model_finetune_path = os.getenv("MODEL_FINETUNE_PATH", None)
    
    # 兼容处理：如果脚本只传了 MODEL_PATH 且没传 FINETUNE_PATH，则将其视为 FINETUNE_PATH
    if model_finetune_path is None and os.getenv("MODEL_PATH"):
        model_finetune_path = os.getenv("MODEL_PATH")

    data_path = os.getenv("TRAIN_DATA_PATH", "/home/fwj/workspace/code/UniTime/UniTime_data/charades/train.json")
    video_folder = os.getenv("VIDEO_FOLDER", "/home/fwj/workspace/VisualSearch/charades/Charades_v1")
    output_dir = os.getenv("OUTPUT_DIR", "./output_rl")
    
    nf_short = int(os.getenv("NF_SHORT", "128"))

    # 1. 加载 Base 模型 (Visual + LLM 基座)
    print(f"Loading Base model from {model_local_path}...")
    loader = LOADERS["qwen2-vl"](
        model_hf_path=model_local_path,
        model_local_path=model_local_path,
        compute_dtype=torch.bfloat16,
        device_map="auto" 
    )
    model, tokenizer, processor, config = loader.load()
    
    # [新增] 加载 LoRA 并合并
    if model_finetune_path:
        print(f"Loading LoRA adapters from {model_finetune_path}...")
        # 这里的 model 已经是加载了权重的 Qwen2VLMRForConditionalGeneration 实例
        model = PeftModel.from_pretrained(model, model_finetune_path)
        
        print("Merging LoRA weights into base model for stable RL training...")
        # 关键步骤：合并权重。这能解决 FSDP 训练时 LoRA 模块可能因为 device_map 切分而导致的各种问题
        model = model.merge_and_unload()
    else:
        print("No SFT finetune path provided, starting RL from Base model directly.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载数据集
    dataset = VideoCentricDataset(
        data_path=data_path,
        video_folder=video_folder,
        fps=2,
        split='train', 
        clip_length=32
    )
    
    # 3. 自定义 Collator
    base_collator = COLLATORS["qwen2-vl"](
        config=config, tokenizer=tokenizer, processor=processor
    )
    
    def rl_collate_fn(batch):
        # 1. 预处理: 移除 batch 中 SFT 的 Assistant 回答
        clean_batch = []
        gt_relevant = []
        gt_windows = []
        modes = []
        
        # ...existing code...
        for item in batch:
            msgs = item['message']
            prompt_msgs = [m for m in msgs if m['role'] != 'assistant']
            
            new_item = item.copy()
            new_item['message'] = prompt_msgs
            clean_batch.append(new_item)
            
            # 收集 GT 信息
            # datasets_mr_relevant.py 中 querys, temporal_windows, relevant 是列表
            # 这里 dataset 取出的 item 里 relevant 是一个列表（针对单个视频可能有多个 query），
            # 但 SFT 数据通常在此处已经被打平成单条。
            # 如果 item['relevant'] 是 list:
            if isinstance(item['relevant'], list):
                 gt_relevant.append(item['relevant'][0])
                 gt_windows.append(item['temporal_window'][0])
            else:
                 gt_relevant.append(item['relevant'])
                 gt_windows.append(item['temporal_window'])

            # 推断 Mode
            vid_mode = 'mr'
            if 'duration' in item:
                 if item['duration'] > nf_short:
                     vid_mode = 'mr_seg'
            if 'mode' in item:
                vid_mode = item['mode']
            
            # 强制逻辑：无论如何, 如果 Dataset 里没有，就根据时长判断
            # 这里必须确保 modes 长度和 batch 长度一致
            modes.append(vid_mode)
        # ...existing code...

        # 2. 调用原始 collator
        collated = base_collator(clean_batch)
        
        # 3. 构造返回
        return {
            "input_ids": collated['input_ids'],
            "attention_mask": collated['attention_mask'],
            "pixel_values_videos": collated.get('pixel_values_videos'),
            "video_grid_thw": collated.get('video_grid_thw'),
            "feature_inputs": collated.get('feature_inputs'),
            "combine_t_list": collated.get('combine_t_list'),
            "relevant": gt_relevant,
            "temporal_window": gt_windows,
            "mode": modes
        }

    # 4. 配置训练参数
    # ...existing code...
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,           
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,            
        max_completion_length=64,     
        max_steps=500,                
        save_steps=100,
        logging_steps=1,
        bf16=True,
        report_to="tensorboard",
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[unified_reward_func],
        args=training_args,
        train_dataset=dataset,
        data_collator=rl_collate_fn,
    )
    
    print("Starting RL Training...")
    trainer.train()
    
    # 保存结果
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()