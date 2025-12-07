"""Metrics utilities such as IoU computation."""

from __future__ import annotations

import torch


def update_confusion_matrix(
    conf_matrix: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    preds = preds.view(-1)
    targets = targets.view(-1)
    mask = targets != ignore_index
    if mask.sum() == 0:
        return
    preds = preds[mask]
    targets = targets[mask]
    indices = targets * num_classes + preds
    hist = torch.bincount(indices, minlength=num_classes ** 2)
    conf_matrix += hist.view(num_classes, num_classes)


def compute_iou(conf_matrix: torch.Tensor):
    intersection = torch.diag(conf_matrix)
    ground_truth = conf_matrix.sum(dim=1)
    predicted = conf_matrix.sum(dim=0)
    union = ground_truth + predicted - intersection
    iou = intersection / union.clamp(min=1.0)
    valid = union > 0
    mean_iou = iou[valid].mean().item() if valid.any() else float("nan")
    return iou, mean_iou
