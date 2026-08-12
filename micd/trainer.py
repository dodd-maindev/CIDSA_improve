"""
Trainer cho khung MICD.

Quản lý:
- Optimizer (SGD, momentum=0.9, poly decay)
- Vòng lặp huấn luyện
- Logging loss từng thành phần
- Cập nhật EMA Teacher mỗi iteration
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .framework import MICDConfig, MICDFramework

logger = logging.getLogger("MICD")


def poly_lr_lambda(epoch: int, max_epochs: int = 300, power: float = 0.9) -> float:
    """Poly decay cho learning rate."""
    return (1.0 - epoch / max(max_epochs, 1)) ** power


class MICDTrainer:
    """
    Trainer cho khung MICD.

    Args:
        model: khung MICDFramework.
        train_loader: DataLoader cho tập train (đã trộn labeled/unlabeled).
        val_loader: DataLoader cho tập validation.
        device: 'cuda' hoặc 'cpu'.
        max_epochs: tổng số epoch.
        lr: learning rate ban đầu.
        momentum: SGD momentum.
        weight_decay: weight decay.
        log_every: log loss sau mỗi N iteration.
        save_path: đường dẫn lưu checkpoint.
    """

    def __init__(
        self,
        model: MICDFramework,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        max_epochs: int = 300,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        log_every: int = 50,
        save_path: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.log_every = log_every
        self.save_path = save_path

        # Optimizer chỉ cho Student (Teacher cập nhật qua EMA)
        params = list(model.student_a.parameters()) + list(model.student_b.parameters())
        self.optimizer = SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: poly_lr_lambda(epoch, max_epochs),
        )

    # ------------------------------------------------------------------ #
    # Một epoch huấn luyện                                                #
    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        # Teacher ở chế độ eval để tính uncertainty chuẩn
        self.model.teacher_a.eval()
        self.model.teacher_b.eval()

        total_losses = {
            "loss_total": 0.0,
            "loss_sup": 0.0,
            "loss_cps": 0.0,
            "loss_con": 0.0,
            "loss_dis": 0.0,
            "n_iters": 0,
        }
        t0 = time.time()

        for it, batch in enumerate(self.train_loader):
            x_lab = batch["image_lab"].to(self.device, non_blocking=True)
            y_lab = batch["label_lab"].to(self.device, non_blocking=True)
            x_unlab = batch.get("image_unlab")
            if x_unlab is not None:
                x_unlab = x_unlab.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(x_lab, y_lab, x_unlab, current_epoch=epoch)
            loss = outputs["loss_total"]
            loss.backward()
            # Clip gradient
            torch.nn.utils.clip_grad_norm_(self.model.student_a.parameters(), max_norm=3.0)
            torch.nn.utils.clip_grad_norm_(self.model.student_b.parameters(), max_norm=3.0)
            self.optimizer.step()

            # Cập nhật Teacher
            self.model.update_teachers()

            # Tích lũy loss
            bs = x_lab.shape[0]
            total_losses["loss_total"] += float(outputs["loss_total"].detach())
            total_losses["loss_sup"] += float(outputs["loss_sup"])
            total_losses["loss_cps"] += float(outputs["loss_cps"])
            total_losses["loss_con"] += float(outputs["loss_con"])
            total_losses["loss_dis"] += float(outputs["loss_dis"])
            total_losses["n_iters"] += 1

            if (it + 1) % self.log_every == 0:
                logger.info(
                    f"Epoch [{epoch}/{self.max_epochs}] Iter [{it + 1}] "
                    f"loss={outputs['loss_total'].item():.4f} "
                    f"sup={outputs['loss_sup'].item():.4f} "
                    f"cps={outputs['loss_cps'].item():.4f} "
                    f"con={outputs['loss_con'].item():.4f} "
                    f"dis={outputs['loss_dis'].item():.4f} "
                    f"β={outputs['beta'].item():.4f}"
                )

        n = max(total_losses["n_iters"], 1)
        avg = {k: v / n for k, v in total_losses.items() if k != "n_iters"}
        elapsed = time.time() - t0
        logger.info(
            f"=== Epoch {epoch} done | "
            f"avg_loss={avg['loss_total']:.4f} | "
            f"time={elapsed:.1f}s"
        )
        return avg

    # ------------------------------------------------------------------ #
    # Vòng lặp huấn luyện                                                  #
    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        logger.info("Bắt đầu huấn luyện MICD...")
        for epoch in range(self.max_epochs):
            self.train_epoch(epoch)
            self.scheduler.step()
            if self.save_path:
                self.save_checkpoint(epoch)

    # ------------------------------------------------------------------ #
    # Lưu / load checkpoint                                               #
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "epoch": epoch,
            "student_a": self.model.student_a.state_dict(),
            "student_b": self.model.student_b.state_dict(),
            "teacher_a": self.model.teacher_a.state_dict(),
            "teacher_b": self.model.teacher_b.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(ckpt, self.save_path)
        logger.info(f"Saved checkpoint → {self.save_path}")