"""End-to-end orchestration for training on the DAVID dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from torch.amp.grad_scaler import GradScaler as TorchGradScaler
except ImportError:  # pragma: no cover - compatibility with older PyTorch
    from torch.cuda.amp import GradScaler as TorchGradScaler  # type: ignore[attr-defined]

from .config import ExperimentConfig
from .data import (
    CLASS_NAMES,
    SegmentationDataset,
    NUM_CLASSES,
    VOID_CLASS_ID,
    compute_class_weights,
    gather_samples,
    load_or_create_splits,
)
from .engine import (
    collate_fn,
    evaluate,
    log_class_iou,
    maybe_visualize,
    save_checkpoint,
    select_device,
    set_seed,
    train_one_epoch,
)
from .model import build_model
import tracemalloc


def _estimate_batch_memory_mb(batch_size: int, image_size: Tuple[int, int]) -> float:
    """Approximate DRAM usage per batch in megabytes."""

    height, width = image_size
    pixels = max(height * width, 1)
    bytes_per_sample = (3 * 4 + 8) * pixels  # float32 image (3ch) + int64 mask
    return batch_size * bytes_per_sample / (1024 ** 2)


def _memory_safe_loader_params(
    batch_size: int,
    image_size: Tuple[int, int],
    requested_workers: int,
    requested_prefetch: Optional[int],
    target_fraction: float = 0.5,
) -> Tuple[int, Optional[int]]:
    """Heuristically cap DataLoader workers/prefetch based on available RAM."""

    if requested_workers <= 0:
        return 0, None

    vm_info = psutil.virtual_memory()
    available_mb = vm_info.available / (1024 ** 2)
    per_batch_mb = max(_estimate_batch_memory_mb(batch_size, image_size), 1.0)
    prefetch = requested_prefetch or 2
    per_worker_mb = per_batch_mb * prefetch
    allowed_mb = max(available_mb * target_fraction, per_batch_mb)
    max_workers = max(1, int(allowed_mb // max(per_worker_mb, 1.0)))
    adjusted_workers = min(requested_workers, max_workers)

    if adjusted_workers < 1:
        adjusted_workers = 1

    # Recompute prefetch if memory still tight
    per_worker_budget = allowed_mb / adjusted_workers
    if per_worker_mb > per_worker_budget:
        new_prefetch = max(int(per_worker_budget // max(per_batch_mb, 1.0)), 1)
    else:
        new_prefetch = prefetch

    return adjusted_workers, new_prefetch


def run_training(config: ExperimentConfig) -> None:
    """
    Orchestrate end-to-end training for DAVID segmentation.

    Args:
        config: ExperimentConfig dataclass with all training parameters.
    """
    set_seed(config.seed)
    device, amp_enabled = select_device(config.backend, config.amp)
    print(f"[Startup] Device: {device.type.upper()} | AMP enabled: {bool(amp_enabled)}")
    # Start tracemalloc if enabled
    if getattr(config, "enable_tracemalloc", False):
        try:
            tracemalloc.start()
            print("[Diagnostics] tracemalloc started for allocation tracking")
        except Exception:
            pass
    # Enable cudnn benchmark if desired
    try:
        import torch.backends.cudnn as cudnn  # type: ignore
        cudnn.benchmark = bool(config.cudnn_benchmark and device.type == "cuda")
    except Exception:
        pass

    splits = load_or_create_splits(config.data_root, config.split, config.force_resplit)
    # If user specified include_sequences, override splits to only use those
    if getattr(config, "include_sequences", None):
        seqs = [s for s in config.include_sequences if s]  # type: ignore[arg-type]
        splits = {"train": seqs, "val": [], "test": []}
    print("Dataset split sizes:", {k: len(v) for k, v in splits.items()})

    train_samples = gather_samples(config.data_root, splits["train"])
    train_weights = compute_class_weights(train_samples, NUM_CLASSES, VOID_CLASS_ID)
    print("Class weights:")
    for idx, weight in enumerate(train_weights.tolist()):
        print(f"  {idx:2d} ({CLASS_NAMES[idx]:>12}): {weight:.3f}")

    train_dataset = SegmentationDataset(
        config.data_root,
        splits["train"],
        config.image_size,
        augment=True,
        diagnostics_interval=getattr(config, "dataset_mem_log_every", 200),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=bool(config.pin_memory and device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=bool(config.persistent_workers and config.num_workers > 0),
        prefetch_factor=(config.prefetch_factor if config.num_workers > 0 else None),
        timeout=config.timeout,
    )

    val_loader = None
    if splits["val"]:
        val_dataset = SegmentationDataset(
            config.data_root,
            splits["val"],
            config.image_size,
            augment=False,
            diagnostics_interval=getattr(config, "dataset_mem_log_every", 200),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=bool(config.pin_memory and device.type == "cuda"),
            collate_fn=collate_fn,
            persistent_workers=bool(config.persistent_workers and config.num_workers > 0),
            prefetch_factor=(config.prefetch_factor if config.num_workers > 0 else None),
            timeout=config.timeout,
        )

    model = build_model(NUM_CLASSES, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=train_weights.to(device),
        ignore_index=VOID_CLASS_ID,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.1,
    )
    scaler = TorchGradScaler(enabled=amp_enabled) if device.type == "cuda" else None

    start_epoch = 1
    best_miou = -math.inf
    # Attach max grad norm to model for clipping in engine
    try:
        setattr(model, "_max_grad_norm", float(config.max_grad_norm))
        setattr(model, "_gc_every_steps", int(getattr(config, "gc_every_steps", 10)))
    except Exception:
        pass
    # Support resume from checkpoint via config
    resume_path = getattr(config, 'resume_from_checkpoint', None)
    if resume_path is not None:
        import os
        if hasattr(resume_path, 'exists'):
            exists = resume_path.exists()
        else:
            exists = os.path.exists(str(resume_path))
        if exists:
            checkpoint = torch.load(str(resume_path), map_location="cpu")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            if scaler is not None and "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_miou = checkpoint.get("best_miou", best_miou)
            print(f"Resumed training from {resume_path} at epoch {start_epoch}")
        else:
            print(f"[WARNING] resume_from_checkpoint is set to '{resume_path}', but file does not exist.")
            response = input("Checkpoint not found. Start training from scratch? [y/N]: ").strip().lower()
            if response != 'y':
                print("Aborting training.")
                return
            print("Proceeding to train from scratch.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.output_dir / "history.json"
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_miou": []}

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\n===== Epoch {epoch}/{config.epochs} =====")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            epoch,
            config.log_interval,
            amp_enabled,
            getattr(config, "batch_mem_log_every", 1),
            getattr(config, "detailed_batch_logging", False),
        )
        history["train_loss"].append(train_loss)

        if epoch % config.val_interval == 0 and val_loader is not None:
            val_loss, val_miou, per_class_iou = evaluate(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled,
            )
            history["val_loss"].append(val_loss)
            history["val_miou"].append(val_miou)
            print(f"Validation loss: {val_loss:.4f}")
            print(f"Validation mIoU: {val_miou:.4f}")
            log_class_iou(per_class_iou)

            is_best = val_miou > best_miou
            if is_best:
                best_miou = val_miou

            checkpoint_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_miou": best_miou,
            }
            save_checkpoint(
                checkpoint_state,
                config.output_dir,
                is_best,
                config.max_checkpoints,
            )

            if epoch % config.visualization_interval == 0:
                viz_dir = config.output_dir / "visualizations" / f"epoch_{epoch:03d}"
                maybe_visualize(
                    model,
                    val_loader,
                    device,
                    amp_enabled,
                    viz_dir,
                )

        scheduler.step()

        # Optional memory relief
        if config.empty_cache_each_epoch and device.type == "cuda":
            torch.cuda.empty_cache()

        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    print(f"Training complete. Best validation mIoU: {best_miou:.4f}")

    import shutil
    # Always save final model checkpoint as best_model.pth
    checkpoint_state = {
        "epoch": config.epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_miou": best_miou,
    }
    save_checkpoint(
        checkpoint_state,
        config.output_dir,
        True,  # is_best
        config.max_checkpoints,
    )
    # Ensure best_model.pth exists
    best_path = config.output_dir / "best_model.pth"
    last_path = config.output_dir / "checkpoint_last.pth"
    # If best_model.pth was not created, copy the last checkpoint
    if not best_path.exists() and last_path.exists():
        shutil.copy(last_path, best_path)
    if best_path.exists():
        print(f"Final model checkpoint saved to {best_path}")
    else:
        print("Warning: Model checkpoint was not saved as best_model.pth!")
