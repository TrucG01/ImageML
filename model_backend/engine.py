"""Training engine utilities for the DAVID pipeline."""

from __future__ import annotations

import psutil
import os

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
    batch_mem_log_every: int,
    detailed_batch_logging: bool,
) -> float:
    """Train the model for one epoch.

    Args:
        model: Model being trained.
        dataloader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Target device.
        scaler: Optional GradScaler for AMP.
        epoch: Current epoch number.
        log_interval: Retained for CLI compatibility.
        amp_enabled: Whether AMP is enabled.
    Returns:
        Average training loss for the epoch.
    """

    log_path = "memory_diagnostics.log"

    def log_mem(msg: str) -> None:
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 ** 2)
        if device.type == "cuda":
            try:
                allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                # Device-level free/total from driver (helps reconcile Task Manager)
                free_b, total_b = torch.cuda.mem_get_info(device)
                free_mb = free_b / (1024 ** 2)
                total_mb = total_b / (1024 ** 2)
            except Exception:
                allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
                try:
                    free_b, total_b = torch.cuda.mem_get_info()
                    free_mb = free_b / (1024 ** 2)
                    total_mb = total_b / (1024 ** 2)
                except Exception:
                    free_mb = 0.0
                    total_mb = 0.0
            vram_str = (
                f"VRAM: {allocated_mb:.2f} MB (alloc), {reserved_mb:.2f} MB (reserved),"
                f" {free_mb:.2f} MB free / {total_mb:.2f} MB total"
            )
        else:
            vram_str = "VRAM: 0.00 MB"
        with open(log_path, "a") as f:
            f.write(f"{msg} | RAM: {ram_mb:.2f} MB | {vram_str}\n")
        print(f"[Diagnostics] {msg} | RAM: {ram_mb:.2f} MB | {vram_str}")

    log_mem(f"Start epoch {epoch}")

    del log_interval  # Legacy argument retained for CLI compatibility

    model.train()
    total_loss = 0.0
    total_batches = 0
    total_steps = len(dataloader)
    # Track DRAM peak (RSS) during the epoch
    process = psutil.Process(os.getpid())
    peak_ram_mb = process.memory_info().rss / (1024 ** 2)
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

    mem_interval = max(1, int(batch_mem_log_every) if batch_mem_log_every else 1)

    for step, (images, targets, _) in enumerate(iterator, start=1):
        should_log = step == 1 or step % mem_interval == 0 or step == total_steps
        stage_prefix = f"Epoch {epoch} Batch {step}"
        if should_log:
            if detailed_batch_logging:
                log_mem(f"{stage_prefix} | stage=dataloader")
            else:
                log_mem(stage_prefix)
            # When detailed logging is requested, append PyTorch allocator summary
            if detailed_batch_logging and device.type == "cuda":
                try:
                    # Record allocator summary and peak stats to help reconcile Task Manager
                    summary = torch.cuda.memory_summary(device=device, abbreviated=False)
                    with open(log_path, 'a') as f:
                        f.write(f"{stage_prefix} CUDA memory summary:\n")
                        f.write(summary + "\n")
                    # Also log max counters
                    with open(log_path, 'a') as f:
                        f.write(
                            f"{stage_prefix} max_allocated={torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB "
                            f"max_reserved={torch.cuda.max_memory_reserved(device) / (1024**2):.2f} MB\n"
                        )
                except Exception:
                    pass
        # Capture a detailed allocation snapshot right after first batch to diagnose RAM spikes
        if step == 1:
            try:
                import tracemalloc
                if tracemalloc.is_tracing():
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')[:30]
                    with open(log_path, 'a') as f:
                        f.write(f"Epoch {epoch} First-batch Top allocations (tracemalloc):\n")
                        for stat in top_stats:
                            f.write(str(stat) + "\n")
            except Exception:
                pass
        # Periodic garbage collection to reclaim Python memory
        if step % getattr(model, "_gc_every_steps", 10) == 0:
            try:
                import gc
                gc.collect()
            except Exception:
                pass
        # Update DRAM peak
        try:
            current_ram_mb = process.memory_info().rss / (1024 ** 2)
            if current_ram_mb > peak_ram_mb:
                peak_ram_mb = current_ram_mb
        except Exception:
            pass
        if detailed_batch_logging and should_log:
            log_mem(f"{stage_prefix} | stage=to_device (before)")
        images = images.to(device)
        targets = targets.to(device)
        if detailed_batch_logging and should_log:
            log_mem(f"{stage_prefix} | stage=to_device (after)")
        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
        )
        with autocast_ctx:
            outputs = model(images)["out"]
            loss = criterion(outputs, targets)
        if detailed_batch_logging and should_log:
            log_mem(f"{stage_prefix} | stage=forward")
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if detailed_batch_logging and should_log:
                log_mem(f"{stage_prefix} | stage=backward_amp")
        else:
            loss.backward()
            # Optional gradient clipping
            try:
                if hasattr(optimizer, "param_groups") and hasattr(loss, "item"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(model, "_max_grad_norm", 0.0) or 0.0)
            except Exception:
                pass
            optimizer.step()
            if detailed_batch_logging and should_log:
                log_mem(f"{stage_prefix} | stage=backward")
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

    log_mem(f"End epoch {epoch}")
    print(f"[Diagnostics] Epoch {epoch} DRAM peak: {peak_ram_mb:.2f} MB")
    # Tracemalloc snapshot for top allocations
    try:
        import tracemalloc
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            with open(log_path, 'a') as f:
                f.write(f"Epoch {epoch} Top allocations (tracemalloc):\n")
                for stat in top_stats:
                    f.write(str(stat) + "\n")
    except Exception:
        pass
    # Log final CUDA allocator summary for the epoch when available
    try:
        if device.type == "cuda":
            with open(log_path, 'a') as f:
                f.write("Epoch end CUDA memory summary:\n")
                f.write(torch.cuda.memory_summary(device=device, abbreviated=False) + "\n")
                f.write(
                    f"Epoch end max_allocated={torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB "
                    f"max_reserved={torch.cuda.max_memory_reserved(device) / (1024**2):.2f} MB\n"
                )
    except Exception:
        pass
    # Optionally free GPU cache to reduce reserved memory between epochs
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
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
    """Evaluate the model on validation data.

    Args:
        model: Model to evaluate.
        dataloader: Validation DataLoader.
        criterion: Loss function for evaluation.
        device: Device used for computation.
        amp_enabled: Whether AMP is enabled.
    Returns:
        Tuple containing average loss, mean IoU, and per-class IoU tensor.
    """

    log_path = "memory_diagnostics.log"

    def log_mem(msg: str) -> None:
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 ** 2)
        if device.type == "cuda":
            try:
                allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
            except Exception:
                allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            vram_str = f"VRAM: {allocated_mb:.2f} MB (alloc), {reserved_mb:.2f} MB (reserved)"
        else:
            vram_str = "VRAM: 0.00 MB"
        with open(log_path, "a") as f:
            f.write(f"{msg} | RAM: {ram_mb:.2f} MB | {vram_str}\n")
        print(f"[Diagnostics] {msg} | RAM: {ram_mb:.2f} MB | {vram_str}")

    log_mem("Start validation")
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
        for step, (images, targets, _) in enumerate(iterator, start=1):
            log_mem(f"Validation Batch {step}")
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
        log_mem("End validation")

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
