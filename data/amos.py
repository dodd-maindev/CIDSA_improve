"""
Dataset loader cho AMOS (Abdominal Multi-Organ Segmentation).

AMOS chứa 360 CT scan với 15 lớp foreground + 1 background.
Cấu trúc thư mục kỳ vọng:
    amos_root/
        imagesTr/
            amos_0001.nii.gz
            amos_0002.nii.gz
            ...
        labelsTr/
            amos_0001.nii.gz
            amos_0002.nii.gz
            ...

Theo paper MICD, tỉ lệ gán nhãn được test: 2%, 5%, 10%.
"""

from __future__ import annotations

import os
import random
from glob import glob
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import get_train_transforms, get_val_transforms


class AMOSDataset(Dataset):
    """
    Dataset bán giám sát cho AMOS.

    Args:
        root: thư mục gốc chứa imagesTr và labelsTr.
        labeled_ratio: tỉ lệ ảnh có nhãn (0.02, 0.05, 0.10).
        roi_size: kích thước crop ngẫu nhiên.
        train: True cho train (có augment), False cho val.
        seed: seed cho việc chia labeled/unlabeled.
    """

    def __init__(
        self,
        root: str,
        labeled_ratio: float = 0.05,
        roi_size: tuple[int, int, int] = (96, 96, 96),
        train: bool = True,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.labeled_ratio = labeled_ratio

        # Lấy danh sách file
        img_dir = self.root / "imagesTr"
        lbl_dir = self.root / "labelsTr"
        all_imgs = sorted(glob(str(img_dir / "*.nii.gz")))

        # Chia labeled / unlabeled
        rng = random.Random(seed)
        rng.shuffle(all_imgs)
        n_labeled = max(1, int(len(all_imgs) * labeled_ratio))

        if train:
            self.labeled_files = all_imgs[:n_labeled]
            self.unlabeled_files = all_imgs[n_labeled:]
        else:
            # Validation: dùng một split riêng (giả sử test cùng tập với train)
            self.labeled_files = all_imgs[:n_labeled]
            self.unlabeled_files = []

        # Tạo transforms
        if train:
            self.tx_labeled = get_train_transforms(roi_size, num_samples=1)
            # Cho unlabeled: chỉ cần load + crop (không cần label)
            from monai.transforms import Compose, LoadImaged, EnsureTyped, RandCropByPosNegLabeld
            self.tx_unlabeled = Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureTyped(keys=["image"]),
                    RandCropByPosNegLabeld(
                        keys=["image"],
                        spatial_size=roi_size,
                        pos=1,
                        neg=1,
                        num_samples=1,
                        image_key="image",
                        allow_smaller=True,
                    ),
                ]
            )
        else:
            self.tx_labeled = get_val_transforms()

    def __len__(self) -> int:
        return max(len(self.labeled_files), len(self.unlabeled_files), 1)

    def _sample_labeled(self) -> dict:
        """Lấy một sample có nhãn."""
        idx = random.randrange(len(self.labeled_files))
        img_path = self.labeled_files[idx]
        lbl_path = img_path.replace("imagesTr", "labelsTr")
        return self.tx_labeled({"image": img_path, "label": lbl_path})[0]

    def _sample_unlabeled(self) -> dict:
        """Lấy một sample không nhãn."""
        if not self.unlabeled_files:
            return None
        idx = random.randrange(len(self.unlabeled_files))
        img_path = self.unlabeled_files[idx]
        return self.tx_unlabeled({"image": img_path})[0]

    def __getitem__(self, idx: int) -> dict:
        """
        Trả về dict chứa:
            - image_lab, label_lab: ảnh và nhãn có nhãn
            - image_unlab: ảnh không nhãn (nếu có)
        """
        sample_lab = self._sample_labeled()
        sample_unlab = self._sample_unlabeled()

        image_lab = sample_lab["image"].unsqueeze(0).float()  # (1, D, H, W)
        label_lab = sample_lab["label"].long()

        out = {
            "image_lab": image_lab,
            "label_lab": label_lab,
        }
        if sample_unlab is not None:
            image_unlab = sample_unlab["image"].unsqueeze(0).float()
            out["image_unlab"] = image_unlab
        return out