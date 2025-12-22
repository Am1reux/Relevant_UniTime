import json
import os
import sys

# 实际可能存在的视频后缀
FILE_FORMATS = ['mp4', 'mkv', 'webm']

def count_unique_videos(json_path):
    """统计去后缀后的唯一视频数"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_videos = set()
    if not isinstance(data, list):
        print(f"{json_path} 根对象不是 list，跳过计数。")
        return unique_videos

    for item in data:
        path = item.get("video_path", "")
        no_ext, _ = os.path.splitext(path)
        unique_videos.add(no_ext)

    print(f"{json_path} 唯一视频数量（去后缀）: {len(unique_videos)}")
    return unique_videos

def fix_video_suffixes(json_path, file_formats=FILE_FORMATS):
    """
    根据实际存在的视频文件修正 video_path 的后缀，
    输出到同目录下的 *_x.json（如 train_x.json, test_x.json）。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"{json_path} 根对象不是 list，无法批量修正 video_path，跳过。")
        return

    changed_cnt = 0
    missing_cnt = 0

    for item in data:
        path = item.get("video_path")
        if not path:
            continue

        dirpath = os.path.dirname(path)
        filename = os.path.basename(path)
        stem, _ = os.path.splitext(filename)

        # 依次尝试各种后缀
        found = False
        for ext in file_formats:
            cand = os.path.join(dirpath, f"{stem}.{ext}")
            if os.path.exists(cand):
                if cand != path:
                    item["video_path"] = cand
                    changed_cnt += 1
                found = True
                break

        if not found:
            missing_cnt += 1

    base, ext = os.path.splitext(json_path)
    out_path = f"{base}_x{ext}"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[fix] {json_path} -> {out_path}, 修正 {changed_cnt} 条，未找到文件 {missing_cnt} 条。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python anet_video_num.py path/to/train.json [path/to/test.json ...]")
        sys.exit(1)

    all_unique = set()
    for jp in sys.argv[1:]:
        # 1) 统计唯一视频数量
        unique = count_unique_videos(jp)
        all_unique |= unique
        # 2) 根据实际文件修正后缀并输出 *_x.json
        fix_video_suffixes(jp)

    if len(sys.argv) > 1:
        print(f"所有输入 json 合并后的唯一视频总数（去后缀）: {len(all_unique)}")