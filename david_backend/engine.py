"""Training engine utilities for the DAVID pipeline."""

from __future__ import annotations

import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:  # optional dependency for nicer progress reporting
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover - tqdm not installed
    tqdm = None  # type: ignore

from .data import CLASS_NAMES, DEFAULT_MEAN, DEFAULT_STD, NUM_CLASSES, VOID_CLASS_ID
from .metrics import compute_iou, update_confusion_matrix
from .visualization import visualize_triplet


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(backend: str, enable_amp: bool) -> Tuple[torch.device, bool]:
    """
    Select device for training (CPU, CUDA, ROCm) and AMP support.

    Args:
        backend: Target backend string ('auto', 'cuda', 'rocm', 'cpu').
        enable_amp: Whether to enable automatic mixed precision.
    Returns:
        Tuple[torch.device, bool]: Device and AMP enabled flag.
    """
    backend = backend.lower()
    if backend not in {"auto", "cuda", "rocm", "cpu"}:
        raise ValueError(f"Unknown backend '{backend}'.")

    if backend == "cpu":
        print("Using CPU backend.")
        return torch.device("cpu"), False

    if not torch.cuda.is_available():
        print(f"Requested backend '{backend}' but no GPU detected. Falling back to CPU.")
        return torch.device("cpu"), False

    hip_version = getattr(torch.version, "hip", None)
    runtime = "ROCm" if hip_version else "CUDA"

    if backend == "rocm" and hip_version is None:
        print("ROCm backend requested but HIP runtime not detected; using CUDA-compatible path.")
    if backend == "cuda" and hip_version is not None:
        print("CUDA backend requested but HIP runtime detected; continuing with ROCm device.")

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    print(f"Using {runtime} device: {device_name}")
    if hip_version:
        print(f"HIP runtime version: {hip_version}")
    return device, enable_amp


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Collate function for DataLoader: stacks images and masks, collects paths.

    Args:
        batch: List of (image, mask, path) tuples.
    Returns:
        Tuple of stacked images, masks, and list of paths.
    """
    images = torch.stack([sample[0] for sample in batch], dim=0)
    masks = torch.stack([sample[1] for sample in batch], dim=0)
    paths = [sample[2] for sample in batch]
    return images, masks, paths


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[Any],
    epoch: int,
    log_interval: int,
    amp_enabled: bool,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Target device.
        scaler: AMP GradScaler or None.
        epoch: Current epoch number.
        log_interval: Logging interval (unused).
        amp_enabled: Whether AMP is enabled.
    Returns:
        float: Average training loss.
    """
    del log_interval  # legacy argument retained for CLI compatibility

    model.train()
    total_loss = 0.0
    total_batches = 0
    total_steps = len(dataloader)
    progress_bar = (
        tqdm(
            dataloader,
            desc=f"Train Epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )
    iterator = progress_bar if progress_bar is not None else dataloader
    use_amp = bool(amp_enabled and scaler is not None and device.type == "cuda")

    for step, (images, targets, _) in enumerate(iterator, start=1):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
        )
        with autocast_ctx:
            outputs = model(images)["out"]
            loss = criterion(outputs, targets)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Optional gradient clipping
            # If optimizer supports param groups, clip across all
            try:
                if hasattr(optimizer, "param_groups") and hasattr(loss, "item"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(model, "_max_grad_norm", 0.0) or 0.0)
            except Exception:
                pass
            optimizer.step()
        total_loss += loss.item()
        total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)

        if progress_bar is not None:
            postfix = {"loss": f"{avg_loss:.4f}"}
            if device.type == "cuda":
                postfix["gpu_mem_gb"] = f"{torch.cuda.memory_allocated(device) / 1e9:.2f}"
            progress_bar.set_postfix(postfix, refresh=False)
        else:
            filled = int(30 * step / max(total_steps, 1))
            bar = "#" * filled + "-" * (30 - filled)
            message = f"\rEpoch {epoch} [{bar}] {step}/{total_steps} Loss {avg_loss:.4f}"
            if device.type == "cuda":
                message += f" GPU Mem {torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
            sys.stdout.write(message)
            sys.stdout.flush()

    if progress_bar is not None:
        progress_bar.close()
    else:
        sys.stdout.write("\n")

    return total_loss / max(total_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float, torch.Tensor]:
    """
    Evaluate the model on validation data.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Target device.
        amp_enabled: Whether AMP is enabled.
    Returns:
        Tuple[float, float, torch.Tensor]: (avg_loss, mean_iou, per_class_iou)
    """
    model.eval()
    total_loss = 0.0
    num_samples: int = 0
    conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)
    progress_bar = (
        tqdm(
            dataloader,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )
    iterator = progress_bar if progress_bar is not None else dataloader
    use_amp = bool(amp_enabled and device.type == "cuda")

    with torch.no_grad():
        for images, targets, _ in iterator:
            images = images.to(device)
            targets = targets.to(device)
            autocast_ctx = (
                torch.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
            )
            with autocast_ctx:
                outputs = model(images)["out"]
                loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            preds = outputs.argmax(dim=1)
            update_confusion_matrix(conf_matrix, preds, targets, NUM_CLASSES, VOID_CLASS_ID)
            if progress_bar is not None:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=False)

    if progress_bar is not None:
        progress_bar.close()

    avg_loss = total_loss / max(num_samples, 1)
    per_class_iou, mean_iou = compute_iou(conf_matrix.cpu())
    return avg_loss, mean_iou, per_class_iou


def save_checkpoint(
    state: Dict[str, torch.Tensor],
    output_dir: Path,
    is_best: bool,
    max_checkpoints: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_epoch{state['epoch']:03d}.pth"
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, output_dir / "best_model.pth")
    checkpoints = sorted(output_dir.glob("checkpoint_epoch*.pth"))
    while len(checkpoints) > max_checkpoints:
        oldest = checkpoints.pop(0)
        oldest.unlink(missing_ok=True)


def maybe_visualize(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    save_dir: Path,
) -> None:
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            return
        images, targets, paths = batch
        images = images.to(device)
        targets = targets.to(device)
        use_amp = bool(amp_enabled and device.type == "cuda")
        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
        )
        with autocast_ctx:
            outputs = model(images)["out"]
        preds = outputs.argmax(dim=1)
        for idx in range(min(len(paths), 4)):
            save_path = save_dir / f"viz_{Path(paths[idx]).stem}.png"
            visualize_triplet(
                images[idx].cpu(),
                targets[idx].cpu(),
                preds[idx].cpu(),
                save_path,
                DEFAULT_MEAN,
                DEFAULT_STD,
            )


def log_class_iou(per_class_iou: torch.Tensor) -> None:
    for class_id, iou in enumerate(per_class_iou.tolist()):
        print(f"  {class_id:2d} ({CLASS_NAMES[class_id]:>12}): {iou:.4f}")
