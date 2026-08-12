"""
Hybrid Swin-UNet 3D Backbone - Cải tiến IMP-1.

Kết hợp:
- Residual 3D UNet encoder/decoder để giữ được inductive bias tốt cho ảnh y khoa
- Swin Transformer block tại bottleneck để bắt giữ long-range context
- Trích xuất 4 feature map decoder cho CFC (tương thích với VNet)

Mục đích: thay thế VNet gốc, giúp cải thiện khả năng phân vùng các cơ quan
nhỏ (túi mật, thực quản, tuyến thượng thận) - vốn yêu cầu ngữ cảnh toàn cục
mà CNN thuần không nắm bắt tốt.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------- #
# Khối CNN cơ bản                                                          #
# ---------------------------------------------------------------------- #
class ResConvBlock(nn.Module):
    """Residual conv block: 2x (Conv3D -> InstanceNorm -> LeakyReLU) + skip."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.leaky_relu(self.norm1(self.conv1(x)), 0.01, inplace=True)
        out = self.norm2(self.conv2(out))
        out = F.leaky_relu(out + identity, 0.01, inplace=True)
        return out


# ---------------------------------------------------------------------- #
# Swin Transformer block 3D (đơn giản)                                    #
# ---------------------------------------------------------------------- #
class WindowAttention3D(nn.Module):
    """Window-based multi-head self-attention 3D (đơn giản hóa).

    Chia volume 3D thành các cửa sổ (W x W x W), tính attention trong mỗi
    cửa sổ. Đây là phiên bản đơn giản hóa để giảm chi phí tính toán.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, window_size ** 3, window_size ** 3)
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape
        ws = self.window_size
        # Pad nếu cần
        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_d or pad_h or pad_w:
            x = F.pad(x.permute(0, 4, 1, 2, 3), (0, pad_w, 0, pad_h, 0, pad_d))
            x = x.permute(0, 2, 3, 4, 1)
            D_p, H_p, W_p = D + pad_d, H + pad_h, W + pad_w
        else:
            D_p, H_p, W_p = D, H, W

        # Chia cửa sổ
        x = rearrange(
            x,
            "B (Dw ws1) (Hw ws2) (Ww ws3) C -> B Dw Hw Ww (ws1 ws2 ws3) C",
            ws1=ws,
            ws2=ws,
            ws3=ws,
        )
        # Attention
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "B ... (h d) -> B ... h d", h=self.num_heads),
            qkv,
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.relative_bias.unsqueeze(0).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, "B ... h d -> B ... (h d)")
        out = self.proj(out)

        # Gộp ngược lại
        out = rearrange(
            out,
            "B Dw Hw Ww (ws1 ws2 ws3) C -> B (Dw ws1) (Hw ws2) (Ww ws3) C",
            ws1=ws,
            ws2=ws,
            ws3=ws,
        )
        # Cắt phần pad
        out = out[:, :D, :H, :W, :].contiguous()
        return out


class SwinBlock3D(nn.Module):
    """Swin Transformer block 3D: Window Attention + MLP, có residual."""

    def __init__(
        self,
        dim: int,
        window_size: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------- #
# Down/Up sampling                                                         #
# ---------------------------------------------------------------------- #
class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.downsample = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.downsample(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=False
            )
        x = x + skip
        return self.block(x)


# ---------------------------------------------------------------------- #
# Hybrid Swin-UNet 3D đầy đủ                                              #
# ---------------------------------------------------------------------- #
class HybridSwinUNet3D(nn.Module):
    """
    Hybrid Swin-UNet 3D.

    Encoder: 4 tầng Residual Conv (down-sampling).
    Bottleneck: 2 khối Swin Transformer 3D để bắt giữ ngữ cảnh toàn cục.
    Decoder: 4 tầng Residual Conv (up-sampling) với skip connection.

    Trả về logits và 4 feature decoder (giống interface VNet).
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 16,
        base_channels: int = 32,
        swin_dim: int = 256,
        swin_blocks: int = 2,
        window_size: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        ch = base_channels

        # Encoder
        self.in_conv = ResConvBlock(in_channels, ch)
        self.down1 = Down(ch, ch * 2)        # ch -> 2ch
        self.down2 = Down(ch * 2, ch * 4)    # 2ch -> 4ch
        self.down3 = Down(ch * 4, ch * 8)    # 4ch -> 8ch
        self.down4 = Down(ch * 8, swin_dim)  # 8ch -> swin_dim

        # Bottleneck Swin Transformer
        self.swin_blocks = nn.ModuleList(
            [SwinBlock3D(swin_dim, window_size, num_heads) for _ in range(swin_blocks)]
        )
        self.swin_norm = nn.LayerNorm(swin_dim)

        # Decoder
        self.up3 = Up(swin_dim, ch * 8)
        self.up2 = Up(ch * 8, ch * 4)
        self.up1 = Up(ch * 4, ch * 2)
        self.up0 = Up(ch * 2, ch)

        self.out_conv = nn.Conv3d(ch, n_classes, kernel_size=1)
        self.feat_channels = [ch * 8, ch * 4, ch * 2, ch]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Encoder
        s0 = self.in_conv(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)  # shape (B, swin_dim, D, H, W)

        # Bottleneck Swin: chuyển sang (B, D, H, W, C)
        B, C, D, H, W = s4.shape
        swin_in = s4.permute(0, 2, 3, 4, 1)
        for blk in self.swin_blocks:
            swin_in = blk(swin_in)
        swin_in = self.swin_norm(swin_in)
        swin_out = swin_in.permute(0, 4, 1, 2, 3).contiguous()

        # Decoder với skip connection
        d3 = self.up3(swin_out, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        decoder_feats: List[torch.Tensor] = [d3, d2, d1, d0]
        logits = self.out_conv(d0)
        return logits, decoder_feats