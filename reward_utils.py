import re
import math

def extract_timestamps(text):
    """
    通用函数：从文本中提取所有浮点数时间戳。
    例如: "Yes, 10.5, 20.0" -> [10.5, 20.0]
          "From 10s to 20s" -> [10.0, 20.0]
    """
    matches = re.findall(r"(\d+(?:\.\d+)?)", text)
    return [float(m[0]) for m in matches]

def compute_iou(box1, box2):
    """
    计算两个 [start, end] 区间的 IoU
    """
    s1, e1 = box1
    s2, e2 = box2
    
    inter_s = max(s1, s2)
    inter_e = min(e1, e2)
    intersection = max(0, inter_e - inter_s)
    
    union = (e1 - s1) + (e2 - s2) - intersection
    
    if union > 0:
        return intersection / union
    return 0.0

def compute_mr_reward(pred_timestamps, gt_windows):
    """
    MR 模式：标准 IoU。
    如果预测了多段，这里简化为与任意 GT 的最大 IoU。
    """
    if len(pred_timestamps) < 2: return 0.0 # 格式错误，没预测出两个点
    
    # 假设前两个数字是 start, end
    pred_box = [pred_timestamps[0], pred_timestamps[1]]
    
    max_iou = 0.0
    for gt in gt_windows:
        # 兼容 gt 可能是 [[s,e]] 嵌套列表
        if isinstance(gt[0], list): gt_box = gt[0]
        else: gt_box = gt
        
        iou = compute_iou(pred_box, gt_box)
        max_iou = max(max_iou, iou)
        
    return max_iou

def compute_mr_seg_reward(pred_timestamps, gt_windows):
    """
    MR_Seg 模式：Min-Max 外包框 IoU。
    完美模拟推理时的粗定位逻辑。
    """
    if not pred_timestamps: return 0.0
    
    # 1. 构建预测外包框
    # 按照推理逻辑，取 min 和 max
    p_min = min(pred_timestamps)
    p_max = max(pred_timestamps)
    
    # 增加 buffer (模拟推理时的扩张 / 防止单点宽度为0)
    buffer = 1.0 
    pred_box = [max(0, p_min - buffer), p_max + buffer]
    
    # 2. 构建 GT 全局外包框
    gt_starts = []
    gt_ends = []
    for gt in gt_windows:
        if isinstance(gt[0], list): s, e = gt[0]
        else: s, e = gt
        gt_starts.append(s)
        gt_ends.append(e)
        
    if not gt_starts: return 0.0
    
    gt_global_box = [min(gt_starts), max(gt_ends)]
    
    # 3. 计算两者 IoU
    return compute_iou(pred_box, gt_global_box)

def unified_reward_func(completions, relevant, temporal_window, mode, **kwargs):
    """
    统一奖励函数入口
    """
    rewards = []
    
    for i, text in enumerate(completions):
        gt_rel = relevant[i]
        gt_wins = temporal_window[i]
        # 处理 batch 中 mode 可能不一致的情况 (虽然通常是一个 batch 一种 mode)
        current_mode = mode[i] if isinstance(mode, list) else mode
        
        # --- [1] 分类奖励 ---
        # 你的逻辑: 负样本包含 "no relevance" (忽略大小写)
        is_pred_neg = bool(re.search(r"no\s+relevance", text, re.IGNORECASE))
        is_pred_pos = not is_pred_neg
        
        rel_score = 0.0
        if is_pred_pos == gt_rel:
            rel_score = 1.0 # 分类正确
        else:
            rewards.append(-1.0) # 分类错误，直接 -1 并退出
            continue
            
        # --- [2] 定位奖励 (仅 TP 样本) ---
        loc_score = 0.0
        if gt_rel: # 只有相关样本才计算 IoU
            pred_times = extract_timestamps(text)
            
            if current_mode == 'mr':
                loc_score = compute_mr_reward(pred_times, gt_wins)
            elif current_mode == 'mr_seg':
                loc_score = compute_mr_seg_reward(pred_times, gt_wins)
        
        # 总分加权: alpha * IoU
        # 分类分保底 + 定位分奖励
        # 示例: alpha = 2.0，最大奖励 = 1.0 + 2.0 = 3.0
        alpha = 2.0
        total_score = rel_score + alpha * loc_score
        rewards.append(total_score)
        
    return rewards