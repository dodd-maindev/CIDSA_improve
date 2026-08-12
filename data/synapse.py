"""
Dataset loader cho Synapse.

Synapse chứa 30 CT scan với 13 lớp foreground (multi-atlas labeling beyond
cranial vault). Tỉ lệ gán nhãn: 20% và 40% theo paper MICD.

Cấu trúc thư mục kỳ vọng:
    synapse_root/
        imagesTr/
            case_0001.nii.gz
            ...
        labelsTr/
            case_0001.nii.gz
            ...
"""

from __future__ import annotations

import random
from glob import glob
from pathlib import Path

from .transforms import get_train_transforms, get_val_transforms


class SynapseDataset:
    """
    Dataset bán giám sát cho Synapse. Tương tự AMOSDataset, khác cấu trúc thư
    mục và có thể dùng 3-fold validation (paper đề cập triple-fold).
    """

    def __init__(
        self,
        root: str,
        labeled_ratio: float = 0.20,
        roi_size: tuple[int, int, int] = (96, 96, 96),
        train: bool = True,
        seed: int = 42,
        n_fold: int = 1,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.labeled_ratio = labeled_ratio

        img_dir = self.root / "imagesTr"
        lbl_dir = self.root / "labelsTr"
        all_imgs = sorted(glob(str(img_dir / "*.nii.gz")))

        rng = random.Random(seed)
        rng.shuffle(all_imgs)
        n_labeled = max(1, int(len(all_imgs) * labeled_ratio))

        if train:
            self.labeled_files = all_imgs[:n_labeled]
            self.unlabeled_files = all_imgs[n_labeled:]
        else:
            self.labeled_files = all_imgs[:n_labeled]
            self.unlabeled_files = []

        if train:
            self.tx_labeled = get_train_transforms(roi_size, num_samples=1)
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
        idx = random.randrange(len(self.labeled_files))
        img_path = self.labeled_files[idx]
        lbl_path = img_path.replace("imagesTr", "labelsTr")
        return self.tx_labeled({"image": img_path, "label": lbl_path})[0]

    def _sample_unlabeled(self) -> dict:
        if not self.unlabeled_files:
            return None
        idx = random.randrange(len(self.unlabeled_files))
        img_path = self.unlabeled_files[idx]
        return self.tx_unlabeled({"image": img_path})[0]

    def __getitem__(self, idx: int) -> dict:
        sample_lab = self._sample_labeled()
        sample_unlab = self._sample_unlabeled()

        image_lab = sample_lab["image"].unsqueeze(0).float()
        label_lab = sample_lab["label"].long()

        out = {"image_lab": image_lab, "label_lab": label_lab}
        if sample_unlab is not None:
            image_unlab = sample_unlab["image"].unsqueeze(0).float()
            out["image_unlab"] = image_unlab
        return out