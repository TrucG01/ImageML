#!/usr/bin/env python
"""Entry point for training DAVID semantic segmentation from YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model_backend import train_from_config


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for DAVID training entrypoint.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on the DAVID dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entrypoint for DAVID training from YAML config.
    """
    args = parse_args()

    # --- Automatic batch size search ---
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Only search if batch_size is set to "auto"
    if str(config.get('training', {}).get('batch_size', '')).lower() == 'auto':
        print("[Auto] Searching for max batch size...")
        from model_backend.data import SegmentationDataset
        from model_backend.pipeline import select_device
        device, _ = select_device(config['backend']['target'], config['backend'].get('amp', True))
        from pathlib import Path
        train_dataset = SegmentationDataset(
            Path(config['dataset']['root']),
            config['dataset']['include'],
            config['training']['image_size'],
            augment=True,
        )
        import gc
        # --- Auto-scale batch size based on GPU VRAM ---
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(f"Detected GPU: {props.name}, {total_mem_gb:.1f} GB VRAM")
            batch_size = min(64, int(total_mem_gb * 2.5))  # Start with a reasonable guess
        else:
            batch_size = 2

        max_batch = None
        min_batch = 1
        success = False

        # --- Read batch search config from YAML ---
        search_cfg = config.get('training', {}).get('batch_search', {})
        max_attempts = int(search_cfg.get('max_attempts', 10))
        initial_batch = int(search_cfg.get('initial', 32))
        min_batch = int(search_cfg.get('min', 1))

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(f"Detected GPU: {props.name}, {total_mem_gb:.1f} GB VRAM")
        else:
            total_mem_gb = 0

        attempts = 0
        batch_size = initial_batch
        last_successful = min_batch
        last_failed = None
        found_success = False

        print(f"Starting batch size search (max {max_attempts} attempts)...")
        while attempts < max_attempts:
            print(f"Attempt {attempts+1}: Trying batch size {batch_size}")
            try:
                x = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)))
                from model_backend.model import build_model
                model = build_model(13, pretrained=True)
                model.to(device)
                with torch.no_grad():
                    out = model(x[0].to(device))
                last_successful = batch_size
                found_success = True
                # If we haven't failed yet, double batch size
                if last_failed is None:
                    batch_size *= 2
                else:
                    # Binary search: midpoint between last_successful and last_failed
                    if last_failed - last_successful <= 1:
                        print(f"Max batch size found: {last_successful}")
                        config['training']['batch_size'] = last_successful
                        torch.cuda.empty_cache()
                        break
                    batch_size = (last_successful + last_failed) // 2
                del model, x, out
                gc.collect()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    last_failed = batch_size
                    if not found_success:
                        batch_size //= 2
                        if batch_size < min_batch:
                            print("Could not find a valid batch size. Try reducing image size or model complexity.")
                            config['training']['batch_size'] = min_batch
                            break
                    else:
                        # Binary search: midpoint between last_successful and last_failed
                        if last_failed - last_successful <= 1:
                            print(f"Max batch size found: {last_successful}")
                            config['training']['batch_size'] = last_successful
                            torch.cuda.empty_cache()
                            break
                        batch_size = (last_successful + last_failed) // 2
                else:
                    raise
            attempts += 1

        if attempts == max_attempts:
            print(f"Batch size search stopped after {max_attempts} attempts. Best found: {last_successful}")
            config['training']['batch_size'] = last_successful

        # Save updated config
        with open(args.config, 'w') as f:
            yaml.safe_dump(config, f)

    train_from_config(args.config)


if __name__ == "__main__":
    main()
