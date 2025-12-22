# 根据 json 文件中的 video_path 提取 vid，检查对应的特征文件
# feat 路径形如：/raid0/fwj/VisualSearch/unitime_feat/qvhl/${vid}.pt

import os
import json
import argparse

def extract_vid(video_path: str) -> str:
    """
    从类似 /xxx/yyy/j7rJstUseKg_360.0_510.0.mp4 的路径中提取 vid：
    j7rJstUseKg_360.0_510.0
    """
    base = os.path.basename(video_path)
    vid, _ = os.path.splitext(base)
    return vid

def check_feats(json_path: str,
                feat_root: str = "/raid0/fwj/VisualSearch/unitime_feat",
                dataset: str = "qvhl",
                list_missing: bool = True):
    feat_dir = os.path.join(feat_root, dataset)
    print(f"JSON: {json_path}")
    print(f"特征目录: {feat_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    seen_vids = set()
    exist_cnt = 0
    missing_cnt = 0
    missing_list = []

    for item in data:
        # 适配当前 qvhl: 使用字段 video_path
        if "video_path" not in item:
            continue
        video_path = item["video_path"]
        if item["mode"] != "mr_seg":
            continue
        vid = extract_vid(video_path)

        # 避免重复检查同一个 vid
        if vid in seen_vids:
            continue
        seen_vids.add(vid)

        feat_path = os.path.join(feat_dir, f"{vid}.pt")
        if os.path.exists(feat_path):
            exist_cnt += 1
        else:
            missing_cnt += 1
            missing_list.append((vid, feat_path))

    print(f"样本条目数: {total}")
    print(f"去重后 vid 数: {len(seen_vids)}")
    print(f"存在特征: {exist_cnt}")
    print(f"缺少特征: {missing_cnt}")

    # if list_missing and missing_list:
    #     # 只打印缺失的 vid
    #     print("\n缺少特征的 vid 列表:")
    #     for vid, _ in missing_list:
    #         print(vid)

def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 json 中的 video_path 检查对应的 feat 是否存在"
    )
    parser.add_argument(
        "--json",
        type=str,
        default="UniTime_data/qvhl/train.json",
        help="标注 json 路径，默认: UniTime_data/qvhl/train.json",
    )
    parser.add_argument(
        "--feat-root",
        type=str,
        default="/raid0/fwj/VisualSearch/unitime_feat",
        help="特征根目录，默认: /raid0/fwj/VisualSearch/unitime_feat",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="qvhl",
        help="数据集子目录名，默认: qvhl，对应 feat_root/qvhl/${vid}.pt",
    )
    parser.add_argument(
        "--no-list-missing",
        action="store_true",
        help="不打印缺少特征的具体 vid 列表，只输出统计",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    check_feats(
        json_path=args.json,
        feat_root=args.feat_root,
        dataset=args.dataset,
        list_missing=not args.no_list_missing,
    )