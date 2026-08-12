"""
Cấu hình cho thí nghiệm AMOS.

Dataset: 360 CT scans, 15 lớp foreground + 1 background.
Theo paper, các tỉ lệ nhãn: 2%, 5%, 10%.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AMOSConfig:
    root: str = "./data/AMOS"
    labeled_ratio: float = 0.05
    roi_size: tuple = (96, 96, 96)

    # Backbone
    backbone: str = "vnet"  # hoặc "hybrid_swin_unet"
    base_channels: int = 16
    in_channels: int = 1
    n_classes: int = 16  # 15 foreground + 1 background

    # Masking
    mask_ratio: float = 0.5
    patch_size: int = 3
    adaptive_mask: bool = False  # IMP-2

    # Training
    batch_size: int = 2
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    max_epochs: int = 300
    num_workers: int = 4

    # EMA
    ema_alpha: float = 0.99

    # Loss schedule
    rampup_length: int = 50
    beta_max: float = 1.0

    # CFC weights
    dynamic_lambda: bool = False  # IMP-3

    # Random init
    w_diff: float = 1.0
    w_dist: float = 1.0

    # Misc
    seed: int = 42
    save_path: str = "./checkpoints/micd_amos.pth"
    log_every: int = 50