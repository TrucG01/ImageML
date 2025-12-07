"""Data utilities for the DAVID dataset."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

try:  # Pillow >=9.1
    Resampling = Image.Resampling  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - compatibility with older Pillow
    class _Resampling:
        BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]
        NEAREST = Image.NEAREST  # type: ignore[attr-defined]

    Resampling = _Resampling()
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import SplitConfig

CLASS_ID_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    0: (220, 220, 0),
    1: (70, 70, 70),
    2: (190, 153, 153),
    3: (250, 170, 160),
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (157, 234, 50),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (107, 142, 35),
    10: (0, 0, 142),
    11: (102, 102, 156),
    12: (0, 0, 0),
}

CLASS_NAMES: Dict[int, str] = {
    0: "traffic_sign",
    1: "building",
    2: "fence",
    3: "other",
    4: "pedestrian",
    5: "pole",
    6: "road_line",
    7: "road",
    8: "sidewalk",
    9: "vegetation",
    10: "car",
    11: "wall",
    12: "void",
}

COLOR_CODE_TO_CLASS_ID: Dict[int, int] = {
    (r << 16) + (g << 8) + b: idx for idx, (r, g, b) in CLASS_ID_TO_COLOR.items()
}

NUM_CLASSES = len(CLASS_ID_TO_COLOR)
VOID_CLASS_ID = 12

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

IMAGE_DIR_CANDIDATES = ("images", "Images", "rgb", "RGB", "img", "frames")
LABEL_DIR_CANDIDATES = ("labels", "Labels", "masks", "semantic", "gt")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SPLIT_FILENAME = "splits_david.json"


def encode_label_image(label_img: Image.Image) -> np.ndarray:
    """
    Map an RGB semantic mask into class IDs for DAVID segmentation.

    Args:
        label_img: PIL Image in RGB format.
    Returns:
        np.ndarray: Encoded mask of class IDs.
    """
    np_label = np.array(label_img, dtype=np.uint8)
    if np_label.ndim != 3 or np_label.shape[2] != 3:
        raise ValueError("Label image must be RGB")
    code = (
        (np_label[:, :, 0].astype(np.uint32) << 16)
        + (np_label[:, :, 1].astype(np.uint32) << 8)
        + np_label[:, :, 2].astype(np.uint32)
    )
    encoded = np.full(code.shape, fill_value=VOID_CLASS_ID, dtype=np.uint8)
    for color_code, class_id in COLOR_CODE_TO_CLASS_ID.items():
        encoded[code == color_code] = class_id
    return encoded


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """
    Convert class IDs back to RGB colors for visualization.

    Args:
        label_map: 2D array of class IDs.
    Returns:
        np.ndarray: RGB visualization of segmentation mask.
    """
    h, w = label_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_ID_TO_COLOR.items():
        mask = label_map == class_id
        colored[mask] = color
    return colored


def _resolve_dataset_roots(root: Path) -> Tuple[Path, Path]:
    """
    Resolve image and label root directories for the DAVID dataset.

    Args:
        root: Path to dataset root.
    Returns:
        Tuple[Path, Path]: (image_root, label_root)
    """
    lower_images = root / "images"
    lower_labels = root / "labels"
    upper_images = root / "Images"
    upper_labels = root / "Labels"

    if upper_images.is_dir() and upper_labels.is_dir():
        return upper_images, upper_labels
    if lower_images.is_dir() and lower_labels.is_dir():
        return lower_images, lower_labels
    return root, root


def discover_sequences(root: Path) -> List[str]:
    """
    Discover available sequence directories in the dataset root.

    Args:
        root: Path to dataset root.
    Returns:
        List[str]: List of sequence directory names.
    """
    image_root, label_root = _resolve_dataset_roots(root)
    if not image_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {image_root}")
    sequence_dirs: List[str] = []
    for entry in image_root.iterdir():
        if not entry.is_dir():
            continue
        if label_root is not image_root:
            label_seq = label_root / entry.name
            if not label_seq.exists() or not label_seq.is_dir():
                continue
        sequence_dirs.append(entry.name)
    if not sequence_dirs:
        raise RuntimeError(f"No sequence directories found in {image_root}")
    sequence_dirs.sort()
    return sequence_dirs


def resolve_subdirectory(
    sequence_dir: Path, candidates: Sequence[str], *, allow_self: bool = False
) -> Path:
    """
    Find a subdirectory matching candidate names, or fallback to self if allowed.

    Args:
        sequence_dir: Path to sequence directory.
        candidates: Possible subdirectory names.
        allow_self: If True, allow using sequence_dir itself if it contains images.
    Returns:
        Path: Resolved subdirectory path.
    """
    for candidate in candidates:
        candidate_path = sequence_dir / candidate
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate_path
    if allow_self:
        for child in sequence_dir.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                return sequence_dir
    raise RuntimeError(
        f"Could not find any of {candidates} inside {sequence_dir}"
    )


def gather_samples(root: Path, sequence_names: Sequence[str]) -> List[Tuple[Path, Path]]:
    """
    Gather (image_path, label_path) pairs for all frames in given sequences.

    Args:
        root: Dataset root path.
        sequence_names: List of sequence names.
    Returns:
        List[Tuple[Path, Path]]: List of (image, label) pairs.
    """
    image_root, label_root = _resolve_dataset_roots(root)
    samples: List[Tuple[Path, Path]] = []
    for seq_name in sequence_names:
        image_seq_dir = image_root / seq_name
        label_seq_dir = label_root / seq_name
        if not image_seq_dir.exists():
            raise FileNotFoundError(f"Sequence directory missing: {image_seq_dir}")
        if not label_seq_dir.exists():
            raise FileNotFoundError(f"Label directory missing for sequence: {label_seq_dir}")
        image_dir = resolve_subdirectory(image_seq_dir, IMAGE_DIR_CANDIDATES, allow_self=True)
        label_dir = resolve_subdirectory(label_seq_dir, LABEL_DIR_CANDIDATES, allow_self=True)
        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise RuntimeError(f"No image files found for sequence {image_seq_dir}")
        for image_path in image_paths:
            label_path = label_dir / image_path.name
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing label for {image_path.name} under {label_dir}"
                )
            samples.append((image_path, label_path))
    if not samples:
        raise RuntimeError("No samples collected from the provided sequences")
    return samples


def load_or_create_splits(root: Path, split_cfg: SplitConfig, force_resplit: bool) -> Dict[str, List[str]]:
    """
    Load or create train/val/test splits for the dataset.

    Args:
        root: Dataset root path.
        split_cfg: SplitConfig dataclass.
        force_resplit: If True, always recreate splits.
    Returns:
        Dict[str, List[str]]: Mapping of split names to sequence lists.
    """
    split_path = root / SPLIT_FILENAME
    if split_path.exists() and not force_resplit:
        with split_path.open("r", encoding="utf-8") as handle:
            import json

            splits = json.load(handle)
        return {key: splits[key] for key in ("train", "val", "test")}

    sequences = discover_sequences(root)
    requested_total = split_cfg.train_count + split_cfg.val_count + split_cfg.test_count
    if len(sequences) < requested_total:
        raise RuntimeError(
            f"Requested split sizes ({requested_total}) exceed available sequences ({len(sequences)})."
        )

    rng = np.random.default_rng(split_cfg.seed)
    rng.shuffle(sequences)

    train = sequences[: split_cfg.train_count]
    val = sequences[
        split_cfg.train_count : split_cfg.train_count + split_cfg.val_count
    ]
    test = sequences[
        split_cfg.train_count + split_cfg.val_count : split_cfg.train_count
        + split_cfg.val_count
        + split_cfg.test_count
    ]

    splits = {"train": train, "val": val, "test": test}
    with split_path.open("w", encoding="utf-8") as handle:
        import json

        json.dump(splits, handle, indent=2)
    return splits


def compute_class_weights(
    samples: Sequence[Tuple[Path, Path]],
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    """
    Compute class weights for loss balancing based on dataset frequency.

    Args:
        samples: List of (image, label) pairs.
        num_classes: Number of classes.
        ignore_index: Class index to ignore.
    Returns:
        torch.Tensor: Class weights tensor.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, label_path in samples:
        mask = Image.open(label_path).convert("RGB")
        encoded = encode_label_image(mask)
        encoded_flat = torch.from_numpy(encoded.reshape(-1))
        valid = encoded_flat != ignore_index
        if valid.any():
            hist = torch.bincount(encoded_flat[valid], minlength=num_classes).to(torch.float64)
            counts += hist
    total = counts.sum().clamp(min=1.0)
    weights = total / counts.clamp(min=1.0)
    weights[ignore_index] = 0.0
    weights /= weights.mean().clamp(min=1e-6)
    return weights.to(torch.float32)


class SegmentationDataset(Dataset):
    """
    PyTorch dataset wrapper for DAVID sequences.

    Args:
        root: Dataset root path.
        sequence_names: List of sequence names.
        image_size: (height, width) tuple for resizing.
        augment: Whether to apply random augmentations.
        mean: Normalization mean.
        std: Normalization std.
    """

    def __init__(
        self,
        root: Path,
        sequence_names: Sequence[str],
        image_size: Tuple[int, int],
        augment: bool,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
    ) -> None:
        self.root = root
        self.sequence_names = list(sequence_names)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.augment = augment
        self.mean = mean
        self.std = std
        self.samples = gather_samples(root, self.sequence_names)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image to target size using bilinear interpolation.
        """
        size = (self.image_size[1], self.image_size[0])
        return image.resize(size, resample=Resampling.BILINEAR)

    def _resize_mask(self, mask: Image.Image) -> Image.Image:
        """
        Resize mask to target size using nearest neighbor interpolation.
        """
        size = (self.image_size[1], self.image_size[0])
        return mask.resize(size, resample=Resampling.NEAREST)

    def _apply_augmentations(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Apply random horizontal flip and small rotation to image and mask.
        """
        if not self.augment:
            return image, mask
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, resample=Resampling.BILINEAR)
            mask = mask.rotate(angle, resample=Resampling.NEAREST)
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a sample from the dataset: normalized image tensor, encoded mask, and image path.
        """
        image_path, label_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(label_path).convert("RGB")
        image, mask = self._apply_augmentations(image, mask)
        image = self._resize_image(image)
        mask = self._resize_mask(mask)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, list(self.mean), list(self.std))
        encoded_mask = torch.from_numpy(encode_label_image(mask)).long()
        return image_tensor, encoded_mask, str(image_path)
