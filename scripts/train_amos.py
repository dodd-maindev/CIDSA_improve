"""
Script huấn luyện khung MICD trên tập AMOS.

Sử dụng:
    python scripts/train_amos.py \
        --root ./data/AMOS \
        --labeled_ratio 0.05 \
        --backbone vnet \
        --adaptive_mask \
        --dynamic_lambda \
        --epochs 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Thêm thư mục cha vào sys.path để import được các package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from configs.config_amos import AMOSConfig
from data.amos import AMOSDataset
from micd.framework import MICDConfig, MICDFramework
from micd.trainer import MICDTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MICD on AMOS")
    p.add_argument("--root", type=str, default="./data/AMOS")
    p.add_argument("--labeled_ratio", type=float, default=0.05)
    p.add_argument("--backbone", type=str, default="vnet",
                   choices=["vnet", "hybrid_swin_unet"])
    p.add_argument("--adaptive_mask", action="store_true",
                   help="Bật IMP-2: uncertainty-guided adaptive masking")
    p.add_argument("--dynamic_lambda", action="store_true",
                   help="Bật IMP-3: dynamic feature consistency weighting")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--ema_alpha", type=float, default=0.99)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--patch_size", type=int, default=3)
    p.add_argument("--save_path", type=str, default="./checkpoints/micd_amos.pth")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    # ---- Cấu hình ----
    cfg = AMOSConfig(
        root=args.root,
        labeled_ratio=args.labeled_ratio,
        backbone=args.backbone,
        adaptive_mask=args.adaptive_mask,
        dynamic_lambda=args.dynamic_lambda,
        max_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ema_alpha=args.ema_alpha,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        save_path=args.save_path,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    logging.info(f"Config: {cfg}")

    torch.manual_seed(cfg.seed)

    # ---- Dữ liệu ----
    train_set = AMOSDataset(
        root=cfg.root,
        labeled_ratio=cfg.labeled_ratio,
        roi_size=cfg.roi_size,
        train=True,
        seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=False,  # dataset đã tự sample ngẫu nhiên
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ---- Model ----
    model_cfg = MICDConfig(
        in_channels=cfg.in_channels,
        n_classes=cfg.n_classes,
        backbone=cfg.backbone,
        base_channels=cfg.base_channels,
        mask_ratio=cfg.mask_ratio,
        patch_size=cfg.patch_size,
        adaptive_mask=cfg.adaptive_mask,
        ema_alpha=cfg.ema_alpha,
        beta_max=cfg.beta_max,
        rampup_length=cfg.rampup_length,
        dynamic_lambda=cfg.dynamic_lambda,
        w_diff=cfg.w_diff,
        w_dist=cfg.w_dist,
    )
    model = MICDFramework(model_cfg)
    n_params = sum(p.numel() for p in model.student_a.parameters())
    logging.info(f"Student A params: {n_params:,}")

    # ---- Trainer ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = MICDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        device=device,
        max_epochs=cfg.max_epochs,
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        log_every=cfg.log_every,
        save_path=cfg.save_path,
    )
    trainer.fit()


if __name__ == "__main__":
    main()