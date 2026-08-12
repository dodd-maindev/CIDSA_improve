"""
Toán tử tạo mặt nạ che 3D (3D Mask Operator) cho MICD.

Triển khai phương trình (2) trong paper:
    xm = M(r, s) ⊙ x

Trong đó:
- r (mask_ratio): tỉ lệ che (mặc định 0.5)
- s (patch_size): kích thước khối che (mặc định 3x3x3)

Ngoài ra còn hỗ trợ **Uncertainty-guided Adaptive Masking** (Cải tiến IMP-2):
- Sinh mask ưu tiên che các vùng có uncertainty thấp (vùng nền/dễ)
- Giữ nguyên các vùng có uncertainty cao (ranh giới/cơ quan nhỏ)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class MaskOperator:
    """
    Toán tử che ảnh 3D với hai chế độ:
    - random : che ngẫu nhiên theo tỉ lệ r và kích thước patch s (paper gốc).
    - adaptive: che ưu tiên vùng uncertainty thấp (IMP-2).
    """

    def __init__(
        self,
        mask_ratio: float = 0.5,
        patch_size: int = 3,
        mode: str = "random",
        keep_value: float = 0.0,
    ) -> None:
        """
        Args:
            mask_ratio: tỉ lệ phần trăm voxel bị che (0.0 - 1.0).
            patch_size: kích thước khối vuông che (s x s x s).
            mode: "random" hoặc "adaptive" (cần truyền uncertainty map).
            keep_value: giá trị thay thế cho voxel bị che (mặc định 0).
        """
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio phải nằm trong [0,1], nhận {mask_ratio}")
        if patch_size < 1:
            raise ValueError(f"patch_size phải >= 1, nhận {patch_size}")
        if mode not in ("random", "adaptive"):
            raise ValueError(f"mode phải là 'random' hoặc 'adaptive', nhận '{mode}'")
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.mode = mode
        self.keep_value = keep_value

    # ------------------------------------------------------------------ #
    # Random masking (paper gốc - Eq. 2)                                 #
    # ------------------------------------------------------------------ #
    def random_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tạo mask ngẫu nhiên trên ảnh 3D.

        Args:
            x: tensor đầu vào shape (B, C, D, H, W).

        Returns:
            Tensor mask cùng shape với x, giá trị 1 = giữ, 0 = che.
        """
        B, C, D, H, W = x.shape
        s = self.patch_size
        # Sinh mask ở độ phân giải thấp (D/s, H/s, W/s) rồi upsample
        md = max(1, D // s)
        mh = max(1, H // s)
        mw = max(1, W // s)

        # Quyết định bao nhiêu patch sẽ bị che
        total_patches = md * mh * mw
        n_masked = int(total_patches * self.mask_ratio)

        # Sinh mask nhị phân cho từng batch
        mask_low = torch.ones(B, 1, md, mh, mw, device=x.device, dtype=x.dtype)
        for b in range(B):
            if n_masked > 0:
                flat_idx = torch.randperm(total_patches, device=x.device)[:n_masked]
                mask_low[b].view(-1)[flat_idx] = 0.0

        # Upsample mask về kích thước ảnh gốc (nearest để giữ tính khối)
        mask = F.interpolate(mask_low, size=(D, H, W), mode="nearest")
        return mask

    # ------------------------------------------------------------------ #
    # Adaptive masking - IMP-2                                            #
    # ------------------------------------------------------------------ #
    def adaptive_mask(
        self,
        x: torch.Tensor,
        uncertainty_map: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sinh mask ưu tiên che vùng có uncertainty thấp.

        Args:
            x: tensor đầu vào shape (B, C, D, H, W).
            uncertainty_map: tensor shape (B, 1, D, H, W), giá trị [0, 1],
                giá trị cao = vùng khó/rất quan trọng (giữ nguyên).
            temperature: nhiệt độ cho phân phối sampling (càng cao càng "dễ" che vùng dễ).

        Returns:
            Tensor mask shape (B, 1, D, H, W), 1 = giữ, 0 = che.
        """
        B, C, D, H, W = x.shape
        s = self.patch_size
        md = max(1, D // s)
        mh = max(1, H // s)
        mw = max(1, W // s)

        # Pool uncertainty xuống độ phân giải patch
        uncertainty_low = F.avg_pool3d(
            uncertainty_map.float(),
            kernel_size=s,
            stride=s,
        )  # (B, 1, md, mh, mw)
        if uncertainty_low.shape[-3:] != (md, mh, mw):
            uncertainty_low = F.interpolate(
                uncertainty_low, size=(md, mh, mw), mode="trilinear",
                align_corners=False,
            )

        # Xác suất GIỮ voxel tỉ lệ thuận với uncertainty
        # Vùng uncertainty cao -> giữ; uncertainty thấp -> che.
        keep_prob = uncertainty_low / (uncertainty_low.mean(dim=(2, 3, 4), keepdim=True) + 1e-8)
        keep_prob = torch.sigmoid(torch.log(keep_prob + 1e-8) / temperature)
        keep_prob = keep_prob / (keep_prob.max() + 1e-8)  # chuẩn hóa về [0, 1]

        # Sinh mask: giữ lại ~ (1 - mask_ratio) phần voxel có keep_prob cao nhất
        n_keep = int(md * mh * mw * (1.0 - self.mask_ratio))
        mask_low = torch.zeros(B, 1, md, mh, mw, device=x.device, dtype=x.dtype)

        for b in range(B):
            flat_prob = keep_prob[b].view(-1)
            topk_idx = torch.topk(flat_prob, k=n_keep, largest=True).indices
            mask_low[b].view(-1)[topk_idx] = 1.0

        # Upsample mask về kích thước ảnh gốc
        mask = F.interpolate(mask_low, size=(D, H, W), mode="nearest")
        return mask

    # ------------------------------------------------------------------ #
    # API công khai                                                       #
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        x: torch.Tensor,
        uncertainty_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Áp dụng mask lên ảnh đầu vào.

        Args:
            x: tensor đầu vào shape (B, C, D, H, W).
            uncertainty_map: chỉ cần khi mode='adaptive'.

        Returns:
            Tensor đã bị che (mask áp lên input). Phần giữ nguyên,
            phần che được thay bằng `keep_value` (mặc định 0).
        """
        if self.mode == "random":
            mask = self.random_mask(x)
        else:
            if uncertainty_map is None:
                raise ValueError("uncertainty_map bắt buộc khi mode='adaptive'")
            mask = self.adaptive_mask(x, uncertainty_map)

        # Ghép mask vào tất cả channel: (B,1,D,H,W) -> (B,C,D,H,W)
        if mask.shape[1] == 1 and x.shape[1] > 1:
            mask = mask.expand_as(x)

        return x * mask + self.keep_value * (1.0 - mask), mask

    def mask_only(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về mask nhị phân (không áp lên ảnh)."""
        return self.random_mask(x)