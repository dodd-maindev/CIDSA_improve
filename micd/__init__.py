"""
Gói MICD - tái cấu trúc khung Masked Image Consistency & Discrepancy Learning
cho phân vùng ảnh y khoa 3D bán giám sát.

Bao gồm:
- mask_operator: Toán tử che 3D cố định + adaptive (theo uncertainty)
- losses       : MCPCLoss, CFCLoss, CMDLoss, DiceCELoss
- ema          : Cập nhật trọng số EMA cho Teacher
- rampup       : Gaussian ramp-up cho hệ số β
- dynamic_weights: Trọng số λ_k động cho CFC
- uncertainty  : Tính uncertainty map từ 2 Teacher
- backbones    : VNet 3D + Hybrid Swin-UNet 3D (IMP-1)
- framework    : Lớp MICD chính
- trainer      : Vòng lặp huấn luyện
"""

__version__ = "1.0.0"