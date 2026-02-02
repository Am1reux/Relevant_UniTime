import os
import json
from PIL import Image
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset
import random 
import pandas as pd
from collections import defaultdict

class VideoCentricDataset(Dataset):
    """
    Dataset for supervised fine-tuning 
    """

    def __init__(
        self,
        data_path: str,
        video_folder: Optional[str] = None,
        feat_folder: Optional[str] = None,
        fps: int = 2,
        split='train',
        num_clips=32,
        clip_length=-1,
    ) -> None:
        super(VideoCentricDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.video_folder = video_folder
        self.feat_folder = feat_folder
        self.fps = fps
        self.is_text_only = [
            False
            for source in self.list_data_dict
        ]
        self.split = split
        self.num_clips = num_clips
        self.clip_length = clip_length
        
        # 采样负样本
        if self.split == 'train':
            self.query_pool = []
            for source in self.list_data_dict:
                vid = source["id"]
                annos = source["annos"]
                for anno in annos:
                    self.query_pool.append({
                        'query': anno['query'],
                        'vid': vid,
                    })
        
        # 采样负样本V1
        # # 优化负采样: 按 vid 分组存储
        # if self.split == 'train':
        #     self.vid_to_queries = defaultdict(list)
        #     self.all_queries = []
            
        #     for source in self.list_data_dict:
        #         vid = source["id"]
        #         annos = source["annos"]
        #         for anno in annos:
        #             query = anno['query']
        #             self.vid_to_queries[vid].append(query)
        #             self.all_queries.append(query)
            
        #     # 为每个 vid 预计算候选负样本
        #     self.negative_pools = {}
        #     for vid in self.vid_to_queries.keys():
        #         self.negative_pools[vid] = [
        #             q for q in self.all_queries 
        #             if q not in self.vid_to_queries[vid]
        #         ]


    def __len__(self) -> int:
        return len(self.list_data_dict)
    
    def construct_messages_mr_fps(self, video_path, feature_path, fps, querys, temporal_windows, retrieval_segment, retrieval_mode, relevant):
        # if retrieval_mode == 'mr_seg':
        #     message = [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1], 
        #                     "feature": f"{feature_path}", "num_clips": self.num_clips, "clip_length": self.clip_length, "temporal_windows":temporal_windows},
        #                 {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to answer the query based on the video content. If the query is relevant to the video, identify the specific timestamp(s) when the given query appears. If the query is not relevant to the video, answer \'No relevance.\'"}
        #             ]
        #         },
        #     ]

        # 第一个版本的提示词
        # if retrieval_mode == 'mr_seg':
        #     message = [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1]},
        #                 {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to answer the query based on the video content. If the query is relevant to the video, identify the specific timestamp(s) when the given query appears. If the query is not relevant to the video, answer \'No relevance.\'"}
        #             ]
        #         },
        #     ]
        # elif retrieval_mode == 'mr':
        #     message = [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1]},
        #                 {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to answer the query based on the video content. If the query is relevant to the video, identify the temporal window (start and end timestamps) where it occurs. If the query is not relevant to the video, answer \'No relevance.\'"}
        #             ]
        #         },
        #     ]
        

        # 第二个版本的提示词：优化过定位
        if retrieval_mode == 'mr_seg':
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1]},
                        {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to answer the query based on the video content. If the query is relevant to the video, answer in the format: \'Yes, From <start>s to <end>s .\'. If the query is not relevant to the video, answer \'No, from -1s to -1s\'"}
                    ]
                },
            ]
        elif retrieval_mode == 'mr':
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1]},
                        {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to answer the query based on the video content. If the query is relevant to the video, answer in the format: \'Yes, From <start>s to <end>s .\'. If the query is not relevant to the video, answer \'No, from -1s to -1s.\'"}
                    ]
                },
            ]

        for query in querys:
            message.append(
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Query:{query}\nAnswer: "}
                ]
            }
            )
        return message
    
    def sample_negative_query(self, cur_vid, num_samples):
        candidate_queries = [q['query'] for q in self.query_pool if q['vid'] != cur_vid]
        if len(candidate_queries) < num_samples:
            return random.choices(candidate_queries, k=num_samples)
        else:
            return random.sample(candidate_queries, k=num_samples)

    def sample_negative_query(self, cur_vid, num_samples):
        candidate_queries = [q['query'] for q in self.query_pool if q['vid'] != cur_vid]
        if len(candidate_queries) < num_samples:
            return random.choices(candidate_queries, k=num_samples)
        else:
            return random.sample(candidate_queries, k=num_samples)
    # 负采样V1
    # def sample_negative_query(self, cur_vid, num_samples):
    #     candidate_queries = self.negative_pools.get(cur_vid, self.all_queries)
    #     if len(candidate_queries) < num_samples:
    #         return random.choices(candidate_queries, k=num_samples)
    #     else:
    #         return random.sample(candidate_queries, k=num_samples)   

    def __getitem__(self, i) -> Dict[str, List]:
        source = self.list_data_dict[i]
        # 这里的取到第i个的信息
        # 看看训练的时候是否能够取到其他i+1 通过打乱顺序的方式
        
        # 记得推理的时候,我的样本实际上已经构建好了
        #
        qid = source["qid"]
        vid = source["id"]
        annos = source["annos"]
        retrieval_mode = source["mode"]

        video_start = source.get("video_start", 0)
        video_end = source.get("video_end", source["duration"])

        temporal_window = [anno["window"] for anno in annos]
        query = [anno["query"] for anno in annos]
        # True or False 代表有答案 和 无答案
        if self.split == 'train':
            relevant = [True for anno in annos]
            # 采样负样本
            num_pos = len(annos)
            num_neg = num_pos
            if num_neg > 0:
                neg_queries = self.sample_negative_query(vid, num_neg)
                for neg_query in neg_queries:
                    query.append(neg_query)
                    temporal_window.append([[]])  # 空时间戳表示无答案
                    relevant.append(False)
        else:
            relevant = [anno["relevant"] for anno in annos]        
        duration = video_end - video_start

        retrieval_segment = [video_start, video_end]
        
        video_path = source.get("video_path", None)
        if video_path is None:
            if 'tacos' in self.video_folder:
                video_path = os.path.join(self.video_folder,f"{vid}.avi")
            else:
                video_path = os.path.join(self.video_folder,f"{vid}.mp4")

        feature_path = source.get("feature_path", None)
        if self.feat_folder is not None and feature_path is None:
            feature_path = os.path.join(self.feat_folder,f"{vid}.pt")

        message = self.construct_messages_mr_fps(video_path=video_path, feature_path=feature_path, fps=self.fps, querys=query, temporal_windows=temporal_window,
                                                 retrieval_segment=retrieval_segment, retrieval_mode=retrieval_mode, relevant=relevant)

        return {"message":message, "split":self.split, "temporal_window":temporal_window, "mode":retrieval_mode, "relevant": relevant, "qid":qid, "duration":duration}