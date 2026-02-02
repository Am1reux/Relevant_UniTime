import re
import torch

def calculate_iou(pred_window, gt_windows):
    """
    计算预测窗口与所有GT窗口的最大IoU
    pred_window: [start, end]
    gt_windows: List[[start, end]]
    """
    if pred_window[0] == -1: return 0.0
    
    max_iou = 0.0
    p_s, p_e = pred_window
    
    # 防止预测出 end < start 的情况
    if p_e < p_s: return 0.0

    for idx, window in enumerate(gt_windows):
        # 兼容 gt_windows 可能是 [[s, e], ...] 也可能是 [[[s,e]]] 嵌套的情况
        if isinstance(window[0], list): g_s, g_e = window[0]
        else: g_s, g_e = window
            
        inter_s = max(p_s, g_s)
        inter_e = min(p_e, g_e)
        intersection = max(0, inter_e - inter_s)
        union = (p_e - p_s) + (g_e - g_s) - intersection
        
        if union > 0:
            iou = intersection / union
            max_iou = max(max_iou, iou)
            
    return max_iou

def extract_content(text):
    """
    解析模型输出
    期望格式: Yes, From 10.5s to 20s.
    或者: No, From -1s to -1s.
    """
    # 稍微放宽一点正则，允许大小写和少量的空格容错，但结构必须对
    # 捕获组: 1=(Yes|No), 2=Start, 3=End
    pattern = r"(Yes|No),?\s+[Ff]rom\s+(-?[\d\.]+)\s*s?\s+to\s+(-?[\d\.]+)\s*s?"
    match = re.search(pattern, text)
    if match:
        is_pos = match.group(1).lower() == "yes"
        try:
            s_time = float(match.group(2))
            e_time = float(match.group(3))
            return is_pos, [s_time, e_time]
        except:
            return None, None
    return None, None

def relevance_reward_func(completions, relevant, **kwargs):
    """相关性判别奖励: TP/TN给正分, FP/FN给负分"""
    rewards = []
    for text, gt_rel in zip(completions, relevant):
        pred_rel, _ = extract_content(text)
        if pred_rel is None: # 格式错误
            rewards.append(0.0) 
            continue
        
        # 二分类奖励
        if pred_rel == gt_rel:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards

def iou_reward_func(completions, relevant, temporal_window, **kwargs):
    """时序定位奖励: 仅在 TP (True Positive) 时计算 IoU"""
    rewards = []
    for text, gt_rel, gt_wins in zip(completions, relevant, temporal_window):
        pred_rel, pred_win = extract_content(text)
        
        # 只有当 GT是相关 且 预测也是相关 时，才计算IoU
        if gt_rel and pred_rel: 
            iou = calculate_iou(pred_win, gt_wins)
            rewards.append(2.0 * iou) # alpha = 2.0
        else:
            rewards.append(0.0)
    return rewards

def format_reward_func(completions, **kwargs):
    """格式一致性奖励: 符合正则给分"""
    rewards = []
    for text in completions:
        pred_rel, _ = extract_content(text)
        if pred_rel is not None:
            rewards.append(0.5) # 格式正确奖励
        else:
            rewards.append(-0.5) # 格式错误惩罚
    return rewards

def logic_reward_func(completions, **kwargs):
    """逻辑性判别: 惩罚自相矛盾"""
    rewards = []
    for text in completions:
        pred_rel, pred_win = extract_content(text)
        if pred_rel is None:
            rewards.append(0.0)
            continue
            
        is_dummy = (pred_win[0] == -1 and pred_win[1] == -1)
        
        # 矛盾A: 说 No (pred_rel=False)，但给了具体时间 (not is_dummy)
        if (not pred_rel) and (not is_dummy):
            rewards.append(-1.0)
        # 矛盾B: 说 Yes (pred_rel=True)，但给了 -1 (is_dummy)
        elif pred_rel and is_dummy:
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
            
    return rewards