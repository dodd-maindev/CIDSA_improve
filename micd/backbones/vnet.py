"""
VNet 3D - kiến trúc xương sống gốc được sử dụng trong paper MICD.

Triển khai theo bài báo gốc V-Net (Millatari et al., 2016) với chỉnh sửa
nhỏ để có thể trích xuất feature ở 4 tầng decoder (yêu cầu của module CFC).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------- #
# Khối cơ bản                                                             #
# ---------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """Hai lớp Conv3D kèm InstanceNorm + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.01, inplace=True)
        x = F.leaky_relu(self.norm2(self.conv2(x)), 0.01, inplace=True)
        return x


class DownTransition(nn.Module):
    """Down-sample với stride 2 + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpTransition(nn.Module):
    """Up-sample + skip connection + ConvBlock. Trả về feature để dùng cho CFC."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvBlock(out_ch, out_ch)
        # Skip connection 1x1
        self.skip_proj = None  # set lazily trong forward nếu cần

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Nếu kích thước không khớp thì pad (do lệch pooling)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=False
            )
        # Cộng skip (kiểu VNet) - residual gate
        x = x + skip
        return self.block(x)


# ---------------------------------------------------------------------- #
# VNet 3D đầy đủ                                                          #
# ---------------------------------------------------------------------- #
class VNet(nn.Module):
    """
    VNet 3D cho segmentation.

    Trả về:
        logits       : (B, n_classes, D, H, W)
        decoder_feats: list 4 feature maps từ tầng decoder (CFC)
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 16,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        ch = base_channels
        # Encoder
        self.in_block = ConvBlock(in_channels, ch)
        self.down1 = DownTransition(ch, ch * 2)        # 16 -> 32
        self.down2 = DownTransition(ch * 2, ch * 4)    # 32 -> 64
        self.down3 = DownTransition(ch * 4, ch * 8)    # 64 -> 128
        self.down4 = DownTransition(ch * 8, ch * 16)   # 128 -> 256

        # Decoder (4 tầng trả về feature cho CFC)
        self.up3 = UpTransition(ch * 16, ch * 8)       # 256 -> 128
        self.up2 = UpTransition(ch * 8, ch * 4)        # 128 -> 64
        self.up1 = UpTransition(ch * 4, ch * 2)        # 64 -> 32
        self.up0 = UpTransition(ch * 2, ch)            # 32 -> 16

        # Output
        self.out_conv = nn.Conv3d(ch, n_classes, kernel_size=1)
        self.feat_channels = [ch * 8, ch * 4, ch * 2, ch]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s0 = self.in_block(x)              # 16
        s1 = self.down1(s0)                # 32
        s2 = self.down2(s1)                # 64
        s3 = self.down3(s2)                # 128
        s4 = self.down4(s3)                # 256

        d3 = self.up3(s4, s3)              # 128 (deepest decoder)
        d2 = self.up2(d3, s2)              # 64
        d1 = self.up1(d2, s1)              # 32
        d0 = self.up0(d1, s0)              # 16

        decoder_feats: List[torch.Tensor] = [d3, d2, d1, d0]
        logits = self.out_conv(d0)
        return logits, decoder_feats