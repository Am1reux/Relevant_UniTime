import os
import sys
import torch
import transformers
from trl import GRPOTrainer, GRPOConfig
from datasets_mr_relevant import VideoCentricDataset
from loaders import LOADERS
from collators import COLLATORS
from reward_utils import (
    relevance_reward_func, 
    iou_reward_func, 
    format_reward_func, 
    logic_reward_func
)

def main():
    parser = transformers.HfArgumentParser((GRPOConfig))
    # 这里我们借用 GRPOConfig，但也需要解析其他的自定义参数
    # 为简单起见，假设通过环境变量或硬编码配置路径，或者你可以自定义 Argument 类
    # 此处演示核心逻辑
    
    # 模拟参数 (请在 shell脚本中对应修改)
    model_path = os.getenv("MODEL_PATH", "/home/fwj/workspace/pretrain_model/Qwen/Qwen2-VL-2B-Instruct")
    data_path = os.getenv("TRAIN_DATA_PATH", "/home/fwj/workspace/code/UniTime/UniTime_data/charades/train.json")
    video_folder = os.getenv("VIDEO_FOLDER", "/home/fwj/workspace/VisualSearch/charades/Charades_v1")
    output_dir = os.getenv("OUTPUT_DIR", "./output_rl")

    # 1. 加载模型
    print(f"Loading model from {model_path}...")
    loader = LOADERS["qwen2-vl"](
        model_local_path=model_path,
        compute_dtype=torch.bfloat16,
        device_map="auto" # 多卡环境推荐 auto
    )
    model, tokenizer, processor, config = loader.load()
    
    # 确保 pad_token 设置正确
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载数据集
    # 使用你现有的 VideoCentricDataset，但我们需要它处于 'test' 模式或者不包含 Assistant 的回答
    # 因为 RL 需要模型自己生成回答。
    # 这里我们用 split='test' 或者修改 dataset 逻辑让它不返回 assistant 标签
    dataset = VideoCentricDataset(
        data_path=data_path,
        video_folder=video_folder,
        fps=2,
        split='train', # 依然用 train 切片的数据
        clip_length=32
    )
    
    # 3. 自定义 Collator
    # GRPOTrainer 需要 inputs 字典。
    # 这里的关键是：inputs 会被解包传给 model.generate
    # 这里的 collator 逻辑参考你的 qwen2_vl_relevant_mr.py，但去掉了 label 掩码部分
    train_collator = COLLATORS["qwen2-vl"](
        config=config, tokenizer=tokenizer, processor=processor
    )
    
    def rl_collate_fn(batch):
        # 1. 预处理: 移除 batch 中 SFT 的 Assistant 回答 (如果有的话)
        # 你的 Dataset 在 construct_messages_mr_fps 里如果加了 assistant，这里要去掉
        # 假设 construct_messages_mr_fps 生成了 [User+Image, Assistant(Label)]
        # 我们只保留 [User+Image]
        
        clean_batch = []
        gt_relevant = []
        gt_windows = []
        
        for item in batch:
            msgs = item['message']
            # 保留 User 指令 (包含视频/图像)，去掉 Assistant 的 Answer
            # 注意：你的 dataset 中 User 最后一句是 "Query:...\nAnswer: "，这正是我们需要的 Prompt
            # 只要去掉紧接着的 role='assistant' 即可
            prompt_msgs = [m for m in msgs if m['role'] != 'assistant']
            
            # 使用副本构建新的 item
            new_item = item.copy()
            new_item['message'] = prompt_msgs
            clean_batch.append(new_item)
            
            # 收集 GT 信息传递给 Reward Function
            gt_relevant.append(item['relevant'][0]) # 假设每个样本1个query
            gt_windows.append(item['temporal_window'][0])

        # 2. 调用原始 collator 处理视频/文本 padding
        collated = train_collator(clean_batch)
        
        # 3. 构造返回给 GRPOTrainer 的字典
        # GRPOTrainer 会自动处理 prompt 和 completion 的拼接
        # 但对于多模态，我们需要传递像素值等
        return {
            "input_ids": collated['input_ids'],
            "attention_mask": collated['attention_mask'],
            "pixel_values_videos": collated.get('pixel_values_videos'),
            "video_grid_thw": collated.get('video_grid_thw'),
            "feature_inputs": collated.get('feature_inputs'),
            "combine_t_list": collated.get('combine_t_list'),
            
            # 传递给 Reward Function 的额外信息 (必须与 signature 对应)
            "relevant": gt_relevant,
            "temporal_window": gt_windows
        }

    # 4. 配置训练参数
    # 如果显存不够，减小 num_generations (G)
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=2e-6,           # RL 学习率通常比 SFT 低
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1, # 视频数据显存占用大
        gradient_accumulation_steps=8,
        num_generations=4,            # GRPO 的 Group Size (G)，一次生成 G 个回答对比
        max_completion_length=64,     # 生成长度限制
        max_steps=500,                # 训练步数
        save_steps=100,
        logging_steps=1,
        bf16=True,
        report_to="tensorboard",
        use_vllm=False,               # Qwen2-VL 在 vLLM 上支持有限，先用原生 generate
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,
            relevance_reward_func,
            iou_reward_func,
            logic_reward_func
        ],
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