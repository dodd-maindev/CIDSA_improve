"""
Các phép biến đổi (transforms) cho ảnh CT 3D.

Sử dụng MONAI cho các phép augmentation tiêu chuẩn:
- Random crop, random flip, random rotate
- Intensity shift/scale cho ảnh CT
"""

from __future__ import annotations

from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Spacingd,
)


def get_train_transforms(
    roi_size: tuple[int, int, int] = (96, 96, 96),
    num_samples: int = 2,
) -> Compose:
    """
    Biến đổi huấn luyện: load ảnh, crop ngẫu nhiên có tỉ lệ foreground/background,
    augment hình học + cường độ.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                allow_smaller=True,
            ),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            RandRotated(
                keys=["image", "label"],
                range_x=0.1,
                range_y=0.1,
                range_z=0.1,
                prob=0.2,
                mode=("bilinear", "nearest"),
            ),
            RandGaussianNoised(keys="image", prob=0.15, std=0.1),
            RandGaussianSmoothd(keys="image", prob=0.1),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        ]
    )


def get_val_transforms() -> Compose:
    """Biến đổi validation: chỉ load + chuẩn hóa hướng + spacing."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
        ]
    )