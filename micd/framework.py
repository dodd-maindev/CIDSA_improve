"""
Khung huấn luyện MICD chính.

Tổng hợp:
- 2 Student + 2 Teacher (cập nhật EMA)
- Toán tử mask (random hoặc adaptive - IMP-2)
- 3 module loss: MCPC + CFC + CMD
- Tổng loss với β Gaussian ramp-up (Eq. 7)
- Hỗ trợ Dynamic λ_k cho CFC (IMP-3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ema import EMAUpdater, build_teacher_from_student
from .losses import (
    CFCLoss,
    CMDLoss,
    DiceCELoss,
    DynamicLambdaController,
    MCPCLoss,
    compute_uncertainty_map,
    gaussian_rampup,
)
from .mask_operator import MaskOperator


# ---------------------------------------------------------------------- #
# Cấu hình                                                                #
# ---------------------------------------------------------------------- #
@dataclass
class MICDConfig:
    """Cấu hình cho khung MICD."""

    in_channels: int = 1
    n_classes: int = 16

    # Backbone
    backbone: str = "vnet"  # "vnet" hoặc "hybrid_swin_unet"
    base_channels: int = 16

    # Masking
    mask_ratio: float = 0.5
    patch_size: int = 3
    adaptive_mask: bool = False  # IMP-2

    # EMA
    ema_alpha: float = 0.99
    ema_warmup_steps: int = 0

    # Loss weights
    beta_max: float = 1.0
    rampup_length: int = 50

    # CFC weights
    dynamic_lambda: bool = False  # IMP-3
    gamma_dyn: float = 1.0

    # Random init weights (theo paper [15] DHC)
    w_diff: float = 1.0
    w_dist: float = 1.0


# ---------------------------------------------------------------------- #
# Khung MICD                                                               #
# ---------------------------------------------------------------------- #
class MICDFramework(nn.Module):
    """
    Lớp khung MICD tổng hợp: 2 Student + 2 Teacher (EMA) + losses.

    API:
        model = MICDFramework(cfg)
        out = model(x_lab, x_unlab, current_epoch, rampup_length)
        loss = out["loss_total"]
    """

    def __init__(self, cfg: MICDConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Chọn backbone
        if cfg.backbone == "vnet":
            from .backbones import VNet
            student_a = VNet(cfg.in_channels, cfg.n_classes, cfg.base_channels)
            student_b = VNet(cfg.in_channels, cfg.n_classes, cfg.base_channels)
        elif cfg.backbone == "hybrid_swin_unet":
            from .backbones import HybridSwinUNet3D
            student_a = HybridSwinUNet3D(cfg.in_channels, cfg.n_classes, cfg.base_channels)
            student_b = HybridSwinUNet3D(cfg.in_channels, cfg.n_classes, cfg.base_channels)
        else:
            raise ValueError(f"Backbone không hỗ trợ: {cfg.backbone}")

        # Khởi tạo hai Student khác nhau (để đảm bảo diversity)
        self._init_different_weights(student_a, student_b)

        # Tạo Teacher từ Student
        self.student_a = student_a
        self.student_b = student_b
        self.teacher_a = build_teacher_from_student(student_a)
        self.teacher_b = build_teacher_from_student(student_b)

        # EMA updater
        self.ema_a = EMAUpdater(self.teacher_a, self.student_a, cfg.ema_alpha, cfg.ema_warmup_steps)
        self.ema_b = EMAUpdater(self.teacher_b, self.student_b, cfg.ema_alpha, cfg.ema_warmup_steps)

        # Mask operator
        mask_mode = "adaptive" if cfg.adaptive_mask else "random"
        self.mask_op = MaskOperator(
            mask_ratio=cfg.mask_ratio,
            patch_size=cfg.patch_size,
            mode=mask_mode,
        )

        # Losses
        self.loss_sup = DiceCELoss()
        self.loss_cps = MCPCLoss(w_diff=cfg.w_diff, w_dist=cfg.w_dist)
        self.loss_con = CFCLoss(num_layers=4, use_dynamic=cfg.dynamic_lambda)
        self.loss_dis = CMDLoss()

        # Dynamic λ controller (IMP-3)
        if cfg.dynamic_lambda:
            self.dyn_lambda = DynamicLambdaController(
                num_layers=4,
                lambda_init=[0.2 * (k + 1) for k in range(4)],
                gamma=cfg.gamma_dyn,
            )

    # ------------------------------------------------------------------ #
    # Khởi tạo hai Student với trọng số khác biệt (W_diff, W_dist)      #
    # ------------------------------------------------------------------ #
    def _init_different_weights(self, m1: nn.Module, m2: nn.Module) -> None:
        """
        Khởi tạo hai mạng với trọng số khác biệt (paper DHC [15] đề xuất).
        Ở đây đơn giản: m1 dùng default init, m2 khởi tạo lại với seed khác.
        """
        # m1 giữ init mặc định
        for p in m1.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.02)

        # m2 khởi tạo lại với hệ số khác → tạo diversity
        for p in m2.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.05)  # gain khác

    # ------------------------------------------------------------------ #
    # Hàm tiện ích: tạo mask + áp lên input                              #
    # ------------------------------------------------------------------ #
    def _masked_input(
        self,
        x: torch.Tensor,
        uncertainty_map: Optional[torch.Tensor] = None,
    ):
        """Trả về (x_masked, mask)."""
        return self.mask_op(x, uncertainty_map=uncertainty_map)

    # ------------------------------------------------------------------ #
    # Forward huấn luyện                                                  #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x_lab: torch.Tensor,
        y_lab: torch.Tensor,
        x_unlab: torch.Tensor,
        current_epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_lab: ảnh có nhãn (B_l, C, D, H, W)
            y_lab: ground truth (B_l, D, H, W) kiểu long
            x_unlab: ảnh không nhãn (B_u, C, D, H, W)
            current_epoch: epoch hiện tại (cho Gaussian ramp-up β)
        """
        device = x_lab.device
        B_l = x_lab.shape[0]
        B_u = x_unlab.shape[0]

        # Tính uncertainty map từ 2 Teacher (chỉ dùng cho adaptive mask)
        uncertainty = None
        if self.cfg.adaptive_mask and B_u > 0:
            with torch.no_grad():
                # Ghép labeled + unlabeled để ước lượng uncertainty tổng thể
                x_for_unc = torch.cat([x_lab, x_unlab], dim=0)
                logits_a, _ = self.teacher_a(x_for_unc)
                logits_b, _ = self.teacher_b(x_for_unc)
                prob_a = F.softmax(logits_a, dim=1)
                prob_b = F.softmax(logits_b, dim=1)
                uncertainty = compute_uncertainty_map(prob_a, prob_b)
                # Tách riêng phần unlabeled để mask
                uncertainty_unlab = uncertainty[B_l:]
                uncertainty_lab = uncertainty[:B_l]

        # Mask ảnh
        if B_u > 0:
            x_unlab_masked, _ = self._masked_input(x_unlab, uncertainty_unlab)
        else:
            x_unlab_masked = x_unlab
        x_lab_masked, _ = self._masked_input(x_lab, uncertainty_lab if uncertainty is not None else None)

        # ----- Forward Student trên ảnh GỐC (cho supervised loss + pseudo-label) ----- #
        logits_a_clean, feats_a_clean = self.student_a(x_lab)
        logits_b_clean, feats_b_clean = self.student_b(x_lab)

        # ----- Forward Student trên ảnh CHE ----- #
        logits_a_mask, feats_a_mask = self.student_a(x_lab_masked)
        logits_b_mask, feats_b_mask = self.student_b(x_lab_masked)

        # ----- Forward Student trên ảnh unlabeled (GỐC + CHE) ----- #
        if B_u > 0:
            logits_a_unlab, feats_a_unlab = self.student_a(x_unlab)
            logits_b_unlab, feats_b_unlab = self.student_b(x_unlab)
            logits_a_unlab_mask, feats_a_unlab_mask = self.student_a(x_unlab_masked)
            logits_b_unlab_mask, feats_b_unlab_mask = self.student_b(x_unlab_masked)

            # Pseudo-label từ ảnh gốc (cho cả MCPC và CMD)
            with torch.no_grad():
                pl_a = logits_a_unlab.argmax(dim=1)
                pl_b = logits_b_unlab.argmax(dim=1)

            # Ghép labeled + unlabeled để tính loss nhất quán
            # Supervised: chỉ trên labeled
            # MCPC + CFC + CMD: trên cả labeled + unlabeled

            # ---- 1. SUPERVISED LOSS (Eq. 6) ----
            L_sup = self.loss_sup(logits_a_clean, y_lab) + self.loss_sup(
                logits_b_clean, y_lab
            )

            # ---- 2. MCPC LOSS (Eq. 3) ----
            # Ghép logits_mask của labeled + unlabeled
            pa_mask_all = torch.cat([logits_a_mask, logits_a_unlab_mask], dim=0)
            pb_mask_all = torch.cat([logits_b_mask, logits_b_unlab_mask], dim=0)
            # Pseudo-label từ đầu ra GỐC (của cả labeled + unlabeled)
            pl_a_clean = logits_a_clean.argmax(dim=1)
            pl_b_clean = logits_b_clean.argmax(dim=1)
            pl_a_unlab_clean = logits_a_unlab.argmax(dim=1)
            pl_b_unlab_clean = logits_b_unlab.argmax(dim=1)
            pl_a_all = torch.cat([pl_a_clean, pl_a_unlab_clean], dim=0)
            pl_b_all = torch.cat([pl_b_clean, pl_b_unlab_clean], dim=0)

            L_cps = self.loss_cps(pa_mask_all, pb_mask_all, pl_a_all, pl_b_all)

            # ---- 3. CFC LOSS (Eq. 4) ----
            feats_a_all = [torch.cat([fa, fu], dim=0) for fa, fu in zip(feats_a_mask, feats_a_unlab_mask)]
            feats_b_all = [torch.cat([fb, fu], dim=0) for fb, fu in zip(feats_b_mask, feats_b_unlab_mask)]

            layer_losses = []
            for k in range(len(feats_a_all)):
                kl_k = self.loss_con._kl_aligned(feats_a_all[k], feats_b_all[k])
                layer_losses.append(float(kl_k.detach()))

            dyn_w = None
            if self.cfg.dynamic_lambda:
                dyn_w = self.dyn_lambda.step(layer_losses)
            L_con = self.loss_con(feats_a_all, feats_b_all, dynamic_weights=dyn_w)

            # ---- 4. CMD LOSS (Eq. 5) ----
            with torch.no_grad():
                logits_a_teach, _ = self.teacher_a(x_unlab)
                logits_b_teach, _ = self.teacher_b(x_unlab)
                pl_a_teach = logits_a_teach.argmax(dim=1)
                pl_b_teach = logits_b_teach.argmax(dim=1)
            L_dis = self.loss_dis(
                logits_a_unlab_mask,
                logits_b_unlab_mask,
                pl_a_teach,
                pl_b_teach,
            )

            # ---- 5. TỔNG LOSS (Eq. 7) ----
            beta = gaussian_rampup(current_epoch, self.cfg.rampup_length)
            beta = min(beta, self.cfg.beta_max)
            L_total = L_sup + self.cfg.beta_max * beta * (L_cps + L_con + L_dis)
        else:
            # Chỉ labeled (warmup phase)
            L_sup = self.loss_sup(logits_a_clean, y_lab) + self.loss_sup(
                logits_b_clean, y_lab
            )
            L_cps = torch.tensor(0.0, device=device)
            L_con = torch.tensor(0.0, device=device)
            L_dis = torch.tensor(0.0, device=device)
            L_total = L_sup
            beta = 0.0

        return {
            "loss_total": L_total,
            "loss_sup": L_sup.detach(),
            "loss_cps": L_cps.detach() if isinstance(L_cps, torch.Tensor) else L_cps,
            "loss_con": L_con.detach() if isinstance(L_con, torch.Tensor) else L_con,
            "loss_dis": L_dis.detach() if isinstance(L_dis, torch.Tensor) else L_dis,
            "beta": torch.tensor(beta, device=device),
        }

    # ------------------------------------------------------------------ #
    # Cập nhật Teacher (EMA)                                              #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_teachers(self) -> None:
        self.ema_a.update()
        self.ema_b.update()