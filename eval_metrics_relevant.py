import argparse
import json
import numpy as np
import os
import copy

def to_window_list_pred_vr(pred):
    windows = np.array(list(filter(lambda x: x != -100, pred)))
    window_list = windows.astype(str).tolist()
    window_list = [float(num) for num in window_list]
    if window_list == []:
        return [-1]
    return window_list

import torch
from nncore.ops import temporal_area, temporal_intersection

def temporal_intersection(windows1, windows2, aligned=False):
    """
    Compute the intersections among temporal windows.

    Args:
        windows1 (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        windows2 (:obj:`nn.Tensor[M, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        aligned (bool, optional): Whether to only compute the intersections
            among aligned temporal windows. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N]` | :obj:`nn.Tensor[N, M]`: The computed \
            intersection values.
    """
    if aligned:
        s = torch.max(windows1[:, 0], windows2[:, 0])
        e = torch.min(windows1[:, 1], windows2[:, 1])
    else:
        s = torch.max(windows1[:, None, 0], windows2[:, 0])
        e = torch.min(windows1[:, None, 1], windows2[:, 1])

    inter = (e - s).clamp(0)
    return inter


def compute_iou_multi(pred, span):
    '''
    添加函数功能
    1 添加判别式的iou计算
    2 传入pred可能是空
    '''
    if (pred == [[-1, -1]] or span == [[-1, -1]]):
        if pred == [[-1, -1]] and span ==[[-1, -1]]:
            return torch.ones(1)
        elif pred == [[-1, -1]] and span != [[-1, -1]]:
            return torch.zeros(1)
        else:
            return torch.zeros(1)

    pred_tensor = torch.Tensor(pred)
    span_tensor = torch.Tensor(span)
    pred_area = temporal_area(pred_tensor).sum()
    span_area = temporal_area(span_tensor).sum()
    # 需要修改
    inter = temporal_intersection(pred_tensor, span_tensor).sum()
    iou = (inter / (pred_area + span_area - inter)).unsqueeze(0)

    iou = torch.where(iou.isfinite(), iou, 0)
    return iou

def compute_relevance_metrics(predictions):
    """
    Compute relevance classification metrics.
    Returns: accuracy, precision, recall, f1
    """
    tp = sum(1 for p in predictions if p["gt_relevant"] and p["pred_relevant"])
    tn = sum(1 for p in predictions if not p["gt_relevant"] and not p["pred_relevant"])
    fp = sum(1 for p in predictions if not p["gt_relevant"] and p["pred_relevant"])
    fn = sum(1 for p in predictions if p["gt_relevant"] and not p["pred_relevant"])
    
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }
def evaluate_performance(
        predictions, thresholds, topK, per_instance=False
):
    """Evalutes the performances."""
    results = [[[] for _ in topK] for _ in thresholds]
    average_IoU = []
    num_instances = 0

    pos_iou_list = []
    neg_iou_list = []

    for pred_datum in predictions:
        i0 = pred_datum["pred_relevant_windows"]
        i1 = [[-1, -1]] if pred_datum["relevant_windows"] == [[]] else pred_datum["relevant_windows"]
        # overlap = compute_iou_multi(pred_datum["pred_relevant_windows"],pred_datum["relevant_windows"]).unsqueeze(0).numpy()
        overlap = compute_iou_multi(i0,i1).unsqueeze(0).numpy()
        average_IoU.append(overlap[0])

        if pred_datum["gt_relevant"]:
            pos_iou_list.append(overlap[0])
        else:
            neg_iou_list.append(overlap[0])

        for tt, threshold in enumerate(thresholds):
            for rr, KK in enumerate(topK):
                results[tt][rr].append((overlap > threshold)[:KK].any())
        num_instances += 1
    mean_results = np.array(results).mean(axis=-1)
    mIoU = np.mean(average_IoU)
    relevance_metrics = compute_relevance_metrics(predictions)

    print(f"Evaluated: {num_instances} instances")
    print(f"  Positive samples: {len(pos_iou_list)}, Negative samples: {len(neg_iou_list)}")
    print(f"Relevance Classification Metrics:")
    print(f"  Accuracy: {relevance_metrics['accuracy']:.4f}")
    print(f"  Precision: {relevance_metrics['precision']:.4f}")
    print(f"  Recall: {relevance_metrics['recall']:.4f}")
    print(f"  F1: {relevance_metrics['f1']:.4f}")
    print(f"  TP: {relevance_metrics['tp']}, TN: {relevance_metrics['tn']}, "
          f"FP: {relevance_metrics['fp']}, FN: {relevance_metrics['fn']}")

    if per_instance:
        per_instance_results = {
            "overlap": overlap,
            "average_IoU": average_IoU,
            "pos_iou": pos_iou_list,
            "neg_iou": neg_iou_list,
            "results": results,
            "relevance_metrics": relevance_metrics,
        }
        return mean_results, mIoU, per_instance_results
    else:
        return mean_results, mIoU, relevance_metrics

def get_metrics(results, mIoU, relevance_metrics, thresholds, topK):
    result_dict = {}
    results *= 100
    mIoU *= 100
    
    # Add temporal grounding metrics
    for ii in range(len(topK)):
        for jj in range(len(thresholds)):
            key = f"Rank@{topK[ii]}\nmIoU@{thresholds[jj]}"
            result_dict[key] = f"{results[jj][ii]:.02f}"

    result_dict["mIoU"] = f"{mIoU:.02f}"
    
    # Add relevance classification metrics
    result_dict["Relevance_Accuracy"] = f"{relevance_metrics['accuracy']*100:.02f}"
    result_dict["Relevance_Precision"] = f"{relevance_metrics['precision']*100:.02f}"
    result_dict["Relevance_Recall"] = f"{relevance_metrics['recall']*100:.02f}"
    result_dict["Relevance_F1"] = f"{relevance_metrics['f1']*100:.02f}"
    
    return result_dict

def save_json(content, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        f.write(json.dumps(content))


def main():
    parser = argparse.ArgumentParser(description='Run Qwen2 - VL inference on Metrics')

    parser.add_argument('--res', type=str, default=None, help='path to results')
    args = parser.parse_args()

    dataset_name_list = ['charades', 'qvhighlights', 'tacos', 'anet', 'ego4d']
    detected_dataset = None
    for dataset in dataset_name_list:
        if dataset in args.res:
            detected_dataset = dataset
            break
    if detected_dataset:
        print(f"Detected dataset: {detected_dataset}")
    else:
        print("No dataset name detected in the data path.")
    
    # [ToModify] path_to_test_data for each benchmark
    gt_paths = {
        "charades":"/home/fwj/workspace/code/UniTime/UniTime_data/charades/test2.0.json",
        "ego4d":"./datasets/ego4d/val_all.json",
        "tacos":"./datasets/tacos/test_all.json",
        "anet":"/home/fwj/workspace/code/UniTime/UniTime_data/anet/test_x.json",
        "qvhighlights":"/home/fwj/workspace/code/UniTime/UniTime_data/qvhl/val.json",
    }
    
    thresholds_dict ={
        "charades": [0.5, 0.7, 0.3],
        "qvhighlights": [0.5, 0.7],
        "ego4d": [0.3, 0.5, 0.7],
        "tacos": [0.3, 0.5, 0.7],    
        "anet": [0.5, 0.7, 0.3],
    }

    results_data = json.load(open(args.res))

    
    gt_path = gt_paths[detected_dataset]
    gt_data = json.load(open(gt_path))
    gt_label = {label["qid"]: label for label in gt_data}

    # results_interpreted_gt = [
    #     {
    #         "qid": int(res['qid']),
    #         "pred_relevant_windows": res["pred_relevant_windows"] if res["pred_relevant_windows"] != [] else [[-1, -1]],
    #         "relevant_windows": gt_label[int(res['qid'])]["annos"][0]["window"],
    #     }
    #     for res in results_data
    # ]
    results_interpreted_gt = [
        {
            "qid": int(res['qid']),
            "pred_relevant": res["pred_relevant"], 
            "pred_relevant_windows": res["pred_relevant_windows"] if res["pred_relevant_windows"] != [] else [[-1, -1]],
            "gt_relevant": gt_label[int(res['qid'])]["annos"][0].get("relevant", True),  # Get from GT
            "relevant_windows": gt_label[int(res['qid'])]["annos"][0]["window"],
        }
        for res in results_data
    ]

    thresholds = thresholds_dict[detected_dataset]
    topK = [1]
    results_gt, mIoU_gt, relevance_metrics = evaluate_performance(
        results_interpreted_gt, thresholds, topK
    )
    
    metrics = get_metrics(results_gt, mIoU_gt, relevance_metrics, thresholds, topK)
    print("\nFinal Metrics:")
    print(metrics)
    
    # Save metrics to file
    metrics_path = os.path.join(os.path.dirname(args.res), "metrics.json")
    save_json(metrics, metrics_path)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()