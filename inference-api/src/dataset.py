from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from typing import Optional

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image

from config import (
    DATA_DIR, IMG_SIZE,
    TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED, CLASS_NAMES,
)


val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class DefectDataset(Dataset):
    """Production defect dataset.

    Scans DATA_DIR for class sub-folders. Folder name is mapped to label by
    CLASS_NAMES index. Unknown folders are skipped.
    """

    def __init__(
        self,
        root_dir: Path = DATA_DIR,
        transform: Optional[object] = None,
    ):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}

        for folder in sorted(root_dir.iterdir()):
            if not folder.is_dir():
                continue
            label = label_map.get(folder.name)
            if label is None:
                continue
            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {root_dir}. "
                "Run scripts/convert_heic_to_jpg.py and scripts/rename_images.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _stratified_split(
    dataset: DefectDataset,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return (train_idx, val_idx, test_idx) using per-class stratified splitting."""
    rng = random.Random(seed)

    label_to_indices: dict[int, list[int]] = {i: [] for i in range(len(CLASS_NAMES))}
    for i, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for indices in label_to_indices.values():
        indices = indices[:]
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def get_test_dataset(root_dir: Path = DATA_DIR) -> Subset:
    """Return the test Subset (used by InferencePipeline)."""
    full_dataset = DefectDataset(root_dir=root_dir, transform=None)
    _, _, test_idx = _stratified_split(
        full_dataset, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED
    )
    test_ds = DefectDataset(root_dir=root_dir, transform=val_transform)
    subset = Subset(test_ds, test_idx)
    subset.indices = test_idx  # type: ignore[attr-defined]
    return subset
