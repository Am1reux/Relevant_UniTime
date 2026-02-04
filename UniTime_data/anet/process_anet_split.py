'''
UniTime_data.anet.process_anet_split 的 Docstring

我会给你提供一个json文件，
格式如下
[
  {
    "qid": 37420,
    "id": "v_nlkmPF8TBdQ",
    "annos": [
      {
        "query": " Next, the cooker add the pasta to the squid and mix, then he and serves in a dish while talking.",
        "window": [
          [
            110.85,
            171.08
          ]
        ]
      }
    ],
    "duration": 174.57,
    "mode": "mr_seg",
    "video_path": "/home/fwj/workspace/VisualSearch/activitynet/v_nlkmPF8TBdQ.mp4"
  },
]
我需要你检查我的video_path是否存在这个文件
如果不存在就需要你进行后缀目录的修改
我的视频路径中肯定存在 /home/fwj/workspace/VisualSearch/activitynet/v_nlkmPF8TBdQ.xxx
具体的后缀不确定，请修改成匹配后的视频后缀 即可
'''


import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _pick_best_match(candidates: List[Path], priority_exts: Tuple[str, ...]) -> Optional[Path]:
    """
    在多个候选文件中挑选最优（按后缀优先级）。
    """
    if not candidates:
        return None
    # 归一化后缀：".mp4" -> "mp4"
    def ext(p: Path) -> str:
        return p.suffix.lower().lstrip(".")

    # 先按优先级排序，再按文件名稳定排序
    prio = {e: i for i, e in enumerate(priority_exts)}
    candidates_sorted = sorted(
        candidates,
        key=lambda p: (prio.get(ext(p), 10**9), p.name)
    )
    return candidates_sorted[0]


def fix_anet_video_paths(
    json_path: str,
    output_path: Optional[str] = None,
    inplace: bool = False,
    priority_exts: Tuple[str, ...] = ("mp4", "m4v", "mkv", "webm", "avi", "mov"),
    strict_same_dir: bool = True,
) -> Dict[str, Any]:
    """
    UniTime_data.anet.process_anet_split 的核心修复函数：检查并修复 ActivityNet 的 video_path 后缀。

    你会提供一个 UniTime 格式的 json 文件（list[dict]），每条样本形如：
    [
      {
        "qid": 37420,
        "id": "v_nlkmPF8TBdQ",
        "annos": [{"query": "...", "window": [[110.85, 171.08]]}],
        "duration": 174.57,
        "mode": "mr_seg",
        "video_path": "/home/fwj/workspace/VisualSearch/activitynet/v_nlkmPF8TBdQ.mp4"
      },
      ...
    ]

    本函数会逐条检查 `video_path` 是否存在：
    - 如果存在：不修改
    - 如果不存在：在同目录下搜索同名文件的其它后缀（例如 v_nlkmPF8TBdQ.*），
      找到后根据后缀优先级（priority_exts）选择一个最合适的文件并替换 video_path。

    典型场景：
    你确信视频一定存在 `/home/fwj/workspace/VisualSearch/activitynet/{video_id}.xxx`，
    但具体后缀不确定（mp4/mkv/webm/...），所以需要自动匹配。

    Args:
        json_path: 输入的 UniTime json 路径。
        output_path: 输出路径（默认 None）。若不提供且 inplace=False，则默认在同目录生成 `*_fixed.json`。
        inplace: 是否原地覆盖写回 json_path。
        priority_exts: 后缀优先级（从高到低），用于候选文件多于1个时选择。
        strict_same_dir: True 表示只在 `video_path` 的父目录里查找同名文件；
                         False 则允许在父目录递归查找（更慢，但更强）。

    Returns:
        一个统计信息 dict，例如：
        {
          "total": 12345,
          "ok": 12000,
          "fixed": 300,
          "missing": 45,
          "fixed_examples": [(old, new), ...最多10条],
          "missing_ids": ["v_xxx", ...最多50条],
          "output_path": "..."
        }

    Raises:
        ValueError: 若输入 json 不是 list，或样本缺少必要字段。
    """
    data = load_json(json_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, got {type(data)}")

    total = len(data)
    ok = 0
    fixed = 0
    missing = 0
    fixed_examples = []
    missing_ids = []

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a dict.")
        if "video_path" not in item or "id" not in item:
            raise ValueError("Each item must contain keys: 'id' and 'video_path'.")

        old_path = Path(item["video_path"])
        if old_path.exists():
            ok += 1
            continue

        # 目标：同目录下找 {stem}.* （stem: 去掉后缀的文件名）
        # 例如 old_path: /.../v_nlkmPF8TBdQ.mp4 -> stem=v_nlkmPF8TBdQ
        stem = old_path.stem
        parent = old_path.parent if old_path.parent.as_posix() != "" else Path(".")

        if strict_same_dir:
            candidates = list(parent.glob(stem + ".*"))
        else:
            # 递归查找：parent/**/stem.*
            candidates = list(parent.rglob(stem + ".*"))

        # 过滤掉目录，只保留文件
        candidates = [p for p in candidates if p.is_file()]

        best = _pick_best_match(candidates, priority_exts)
        if best is None:
            missing += 1
            if len(missing_ids) < 50:
                missing_ids.append(item.get("id", stem))
            continue

        item["video_path"] = str(best)
        fixed += 1
        if len(fixed_examples) < 10:
            fixed_examples.append((str(old_path), str(best)))

    # 输出路径策略
    in_path = Path(json_path)
    if inplace:
        out_path = in_path
    else:
        if output_path is None:
            out_path = in_path.with_name(in_path.stem + "_fixed.json")
        else:
            out_path = Path(output_path)

    save_json(data, str(out_path))

    return {
        "total": total,
        "ok": ok,
        "fixed": fixed,
        "missing": missing,
        "fixed_examples": fixed_examples,
        "missing_ids": missing_ids,
        "output_path": str(out_path),
    }


if __name__ == "__main__":
    # 示例：python process_anet_split.py /path/to/val2.0.json --inplace
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--recursive", action="store_true", help="search recursively under video dir")
    args = parser.parse_args()

    stats = fix_anet_video_paths(
        json_path=args.json_path,
        output_path=args.output,
        inplace=args.inplace,
        strict_same_dir=(not args.recursive),
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
