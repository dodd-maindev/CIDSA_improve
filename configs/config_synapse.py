"""
Cấu hình cho thí nghiệm Synapse.

Dataset: 30 CT scans, 13 lớp foreground.
Theo paper, các tỉ lệ nhãn: 20%, 40%. Triple-fold validation.
"""

from dataclasses import dataclass


@dataclass
class SynapseConfig:
    root: str = "./data/Synapse"
    labeled_ratio: float = 0.20
    roi_size: tuple = (96, 96, 96)
    n_fold: int = 1  # 1/2/3 trong triple-fold validation

    # Backbone
    backbone: str = "vnet"
    base_channels: int = 16
    in_channels: int = 1
    n_classes: int = 14  # 13 foreground + 1 background

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
    save_path: str = "./checkpoints/micd_synapse.pth"
    log_every: int = 50