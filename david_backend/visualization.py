"""Visualization helpers for DAVID segmentation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CLASS_ID_TO_COLOR, colorize_label_map, DEFAULT_MEAN, DEFAULT_STD


def denormalize_image(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std_tensor + mean_tensor


def visualize_triplet(
    image_tensor: torch.Tensor,
    target_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    save_path: Path,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> None:
    image_tensor = denormalize_image(image_tensor.cpu(), mean, std).clamp(0.0, 1.0)
    image_np = (image_tensor.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    gt_np = colorize_label_map(target_mask.cpu().numpy())
    pred_np = colorize_label_map(prediction_mask.cpu().numpy())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[1].imshow(gt_np)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_np)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
