import torch

from .metrics import empty_counts, update_confusion_counts, compute_metrics_from_counts, evaluate_mean_per_image


def _extract_seg_logits(model_output):
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


@torch.no_grad()
def evaluate_micro(model, loader, criterion, device, threshold=0.5):
    """Micro/global Dice+IoU+precision+recall+specificity+accuracy over the whole loader."""
    model.eval()
    total_loss = 0.0
    counts = empty_counts()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        model_output = model(images)
        loss = criterion(model_output, masks)
        total_loss += float(loss.item())

        seg_logits = _extract_seg_logits(model_output)
        probs = torch.sigmoid(seg_logits)
        update_confusion_counts(probs, masks, threshold=threshold, counts=counts)

    metrics = compute_metrics_from_counts(counts)
    metrics["loss"] = total_loss / len(loader)
    metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"] = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    return metrics


@torch.no_grad()
def evaluate_full(model, loader, device, threshold=0.5, criterion=None):
    """Micro metrics (+loss if criterion given) and mean per-image Dice/IoU."""
    if criterion is not None:
        micro = evaluate_micro(model, loader, criterion, device, threshold)
    else:
        model.eval()
        counts = empty_counts()
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            seg_logits = _extract_seg_logits(model(images))
            probs = torch.sigmoid(seg_logits)
            update_confusion_counts(probs, masks, threshold=threshold, counts=counts)
        micro = compute_metrics_from_counts(counts)
        micro["loss"] = float("nan")
        micro["tp"], micro["fp"], micro["tn"], micro["fn"] = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    model.eval()
    all_probs, all_masks = [], []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        seg_logits = _extract_seg_logits(model(images))
        probs = torch.sigmoid(seg_logits)
        all_probs.append(probs.detach().cpu())
        all_masks.append(masks.detach().cpu())

    probs_cat = torch.cat(all_probs, dim=0)
    masks_cat = torch.cat(all_masks, dim=0)
    mean_metrics = evaluate_mean_per_image(probs_cat, masks_cat, threshold=threshold)

    out = {f"micro_{k}": v for k, v in micro.items()}
    out.update(mean_metrics)
    return out
