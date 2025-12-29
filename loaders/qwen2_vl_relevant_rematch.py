from typing import Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from . import register_loader
from .base import BaseModelLoader
from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor

import torch

@register_loader("qwen2-vl")
class Qwen2VLModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        if load_model:
            if self.model_finetune_path is None:
                model = Qwen2VLMRForConditionalGeneration.from_pretrained(
                    self.model_local_path,
                    **self.loading_kwargs,
                ) 
            else:
                model = Qwen2VLMRForConditionalGeneration.from_pretrained(
                    self.model_finetune_path,
                    **self.loading_kwargs,
                ) 
        processor = Qwen2VLMRProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer
        model.tokenizer = tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)

        #  添加特殊token的处理
        # print("原始tokenizer数量", len(tokenizer))
        lrt_token = "<|lrt|>"
        if lrt_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([lrt_token], special_tokens=True)
            # model.resize_token_embeddings(len(tokenizer))
            # lrt_id = tokenizer.convert_tokens_to_ids(lrt_token)
            # print("添加了lrt,tokenizer数量", len(tokenizer), "lrt_token_id", lrt_id)            

            # # 对于 lrt token进行初始化
            # ref_token = "<|im_end|>"
            # ref_id = tokenizer.convert_tokens_to_ids(ref_token)

            # with torch.no_grad():
            #     # 获取 input embeddings 矩阵
            #     input_embeddings = model.get_input_embeddings()
            #     # 复制权重
            #     input_embeddings.weight[lrt_id] = input_embeddings.weight[ref_id].clone()
                
            #     # 如果有 lm_head (output embeddings) 且不共享权重，建议也处理一下
            #     # Qwen2 通常是 tied weights 或者会在 resize 时自动处理，但为了保险： 
            #     output_embeddings = model.get_output_embeddings()
            #     if output_embeddings is not None and output_embeddings.weight.shape[0] == len(tokenizer):
            #         output_embeddings.weight[lrt_id] = output_embeddings.weight[ref_id].clone()


        return model, tokenizer, processor, config