import numpy as np
import torch


def _validate_threshold(threshold: float) -> None:
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")


def empty_counts():
    return {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}


@torch.no_grad()
def update_confusion_counts(probs, target, threshold, counts):
    """Update pixel-level TP/FP/TN/FN counts. probs, target: [B,1,H,W]."""
    _validate_threshold(threshold)
    pred = (probs > threshold).float()

    counts["tp"] += (pred * target).sum().item()
    counts["fp"] += (pred * (1 - target)).sum().item()
    counts["tn"] += ((1 - pred) * (1 - target)).sum().item()
    counts["fn"] += ((1 - pred) * target).sum().item()


def compute_metrics_from_counts(counts, eps=1e-7):
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    return {
        "dice": float(dice), "iou": float(iou), "precision": float(precision),
        "recall": float(recall), "specificity": float(specificity), "accuracy": float(accuracy),
    }


def _to_2d_bool(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {x.shape}")
    return x.astype(bool)


def dice_per_image(pred_bin, gt_bin, eps: float = 1e-7) -> float:
    pred = _to_2d_bool(pred_bin)
    gt = _to_2d_bool(gt_bin)
    inter = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    return float((2.0 * inter + eps) / (denom + eps))


def iou_per_image(pred_bin, gt_bin, eps: float = 1e-7) -> float:
    pred = _to_2d_bool(pred_bin)
    gt = _to_2d_bool(gt_bin)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return float((inter + eps) / (union + eps))


def evaluate_mean_per_image(probs, masks, threshold: float = 0.5) -> dict:
    """Mean/std per-image Dice and IoU. probs, masks: [B,1,H,W] tensors."""
    _validate_threshold(threshold)
    if probs.ndim != 4 or masks.ndim != 4:
        raise ValueError(f"Expected [B,1,H,W], got probs={tuple(probs.shape)}, masks={tuple(masks.shape)}")
    if probs.shape != masks.shape:
        raise ValueError(f"Shape mismatch: probs={tuple(probs.shape)} masks={tuple(masks.shape)}")

    probs_np = probs.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    if masks_np.min() < -1e-6 or masks_np.max() > 1.0 + 1e-6:
        raise ValueError(f"Masks should be in [0,1], got min={masks_np.min():.4f}, max={masks_np.max():.4f}")

    pred_np = probs_np > threshold
    gt_np = masks_np > 0.5

    dice_values, iou_values = [], []
    for i in range(pred_np.shape[0]):
        dice_values.append(dice_per_image(pred_np[i, 0], gt_np[i, 0]))
        iou_values.append(iou_per_image(pred_np[i, 0], gt_np[i, 0]))

    dice_arr = np.asarray(dice_values, dtype=np.float32)
    iou_arr = np.asarray(iou_values, dtype=np.float32)
    return {
        "dice_mean": float(dice_arr.mean()) if dice_arr.size else float("nan"),
        "dice_std": float(dice_arr.std()) if dice_arr.size else float("nan"),
        "iou_mean": float(iou_arr.mean()) if iou_arr.size else float("nan"),
        "iou_std": float(iou_arr.std()) if iou_arr.size else float("nan"),
    }
