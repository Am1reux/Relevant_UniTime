#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

BASE_DIR = "/raid5/hl/VisualSearch/activitynet/VideoData"

def load_video_paths(json_file):
    """
    从 json 文件中读取所有 video_path 字段
    返回：list[str]
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_paths = []

    # 支持两种常见格式：
    # 1) [ {"video_path": "..."} , ... ]
    # 2) { "xxx": {"video_path": "..."}, ... }
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "video_path" in item:
                video_paths.append(item["video_path"])
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict) and "video_path" in v:
                video_paths.append(v["video_path"])
    else:
        print(f"[WARN] {json_file} 的结构不是 list 也不是 dict，跳过。")

    return video_paths


def normalize_path(p):
    """
    规范化路径：
    - 如果是相对路径，则拼到 BASE_DIR 下面
    - 如果是绝对路径但不在 BASE_DIR 下，只是返回原路径（仍然检查是否存在）
    """
    if not os.path.isabs(p):
        return os.path.join(BASE_DIR, p)
    return p


def check_one_json(json_file):
    print(f"\n检查文件: {json_file}")
    video_paths = load_video_paths(json_file)
    print(f"  共读取到 {len(video_paths)} 条 video_path")

    missing = []
    wrong_base = []

    for vp in video_paths:
        full_path = normalize_path(vp)

        # 检查是否在指定目录下
        if not os.path.realpath(full_path).startswith(os.path.realpath(BASE_DIR)):
            wrong_base.append(vp)

        # 检查文件是否存在
        if not os.path.exists(full_path):
            missing.append(vp)

    print(f"  不在目录 {BASE_DIR} 下的路径数量: {len(wrong_base)}")
    print(f"  磁盘上不存在的文件数量:      {len(missing)}")

    if wrong_base:
        print("\n  不在 BASE_DIR 下的示例路径（最多显示前 10 条）：")
        for p in wrong_base[:10]:
            print("   ", p)

    if missing:
        print("\n  找不到的文件路径（最多显示前 20 条）：")
        for p in missing[:20]:
            print("   ", p)

    return missing, wrong_base


def main():
    parser = argparse.ArgumentParser(
        description="检查两个 JSON 文件中的视频路径是否都在指定目录下并且真实存在"
    )
    parser.add_argument("json1", help="第一个 JSON 文件路径")
    parser.add_argument("json2", help="第二个 JSON 文件路径")
    args = parser.parse_args()

    for jf in [args.json1, args.json2]:
        check_one_json(jf)


if __name__ == "__main__":
    main()