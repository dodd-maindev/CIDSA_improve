"""
Các hàm tính toán tổn thất tùy chỉnh cho MICD.

Bao gồm:
- DiceCELoss        : Supervised loss = CE + Dice (Eq. 6)
- MCPCLoss          : Masked Cross Pseudo Consistency (Eq. 3)
- CFCLoss           : Cross Feature Consistency (Eq. 4)
- CMDLoss           : Cross Model Discrepancy (Eq. 5)
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== #
# 1. SUPERVISED LOSS (CE + Dice)                                         #
# ====================================================================== #
class DiceCELoss(nn.Module):
    """
    Tổn thất giám sát: kết hợp Cross Entropy và Dice loss.

    Công thức (Eq. 6):
        L_s(p, y) = L_ce(p, y) + L_dice(p, y)
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-5,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính Dice loss đa lớp.

        Args:
            pred: logits (B, C, D, H, W) hoặc (B, C, H, W).
            target: ground truth (B, D, H, W) hoặc (B, H, W) kiểu long.
        """
        num_classes = pred.shape[1]
        prob = F.softmax(pred, dim=1)

        # One-hot target
        target_oh = F.one_hot(target.clamp_min(0), num_classes=num_classes)
        target_oh = target_oh.permute(0, -1, *range(1, target_oh.ndim - 1)).float()

        # Loại bỏ ignore_index (background tùy chọn)
        dims = tuple(range(2, pred.ndim))
        intersect = (prob * target_oh).sum(dim=dims)
        denom = prob.sum(dim=dims) + target_oh.sum(dim=dims)
        dice = (2.0 * intersect + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: logits (B, C, ...).
            target: long tensor (B, ...) với ignore_index = -1 cho voxel bỏ qua.
        """
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ====================================================================== #
# 2. MCPC LOSS (Eq. 3)                                                   #
# ====================================================================== #
class MCPCLoss(nn.Module):
    """
    Masked Cross Pseudo Consistency Loss.

    Cho ảnh che xm_i đi qua 2 student, lấy pseudo-label từ đầu ra KHÔNG che
    của mỗi nhánh, rồi ép nhánh kia dự đoán khớp pseudo-label chéo.

    L_cps = (1/(N+M)) * Σ [ W_diff * CE(p_A, ŷ_B^m)
                            + W_dist * CE(p_B, ŷ_A^m) ]
    """

    def __init__(
        self,
        w_diff: float = 1.0,
        w_dist: float = 1.0,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.w_diff = w_diff
        self.w_dist = w_dist
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        pred_a_masked: torch.Tensor,
        pred_b_masked: torch.Tensor,
        pseudo_a: torch.Tensor,
        pseudo_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_a_masked: logits của Student A trên ảnh che (B, C, ...).
            pred_b_masked: logits của Student B trên ảnh che (B, C, ...).
            pseudo_a: pseudo-label từ Student A trên ảnh GỐC (B, ...).
            pseudo_b: pseudo-label từ Student B trên ảnh GỐC (B, ...).
        """
        # Cross pseudo: A bị che phải dự đoán theo pseudo-label của B (và ngược lại)
        loss_a = self.ce(pred_a_masked, pseudo_b)
        loss_b = self.ce(pred_b_masked, pseudo_a)
        return self.w_diff * loss_a + self.w_dist * loss_b


# ====================================================================== #
# 3. CFC LOSS (Eq. 4)                                                    #
# ====================================================================== #
class CFCLoss(nn.Module):
    """
    Cross Feature Consistency Loss giữa các đặc trưng decoder của 2 student.

    L_con = (1/(N+M)) * Σ [ Σ_{k=1}^{4} λ_k * KL(d_A^k || d_B^k) ]

    Với trọng số mặc định λ_k = 0.2 * k (tăng dần theo độ sâu).
    Có hỗ trợ trọng số động (Dynamic) truyền từ bên ngoài.
    """

    def __init__(
        self,
        num_layers: int = 4,
        lambda_init: Sequence[float] | None = None,
        use_dynamic: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        if lambda_init is None:
            # Mặc định: [0.2, 0.4, 0.6, 0.8]
            lambda_init = [0.2 * (k + 1) for k in range(num_layers)]
        if len(lambda_init) != num_layers:
            raise ValueError(
                f"lambda_init phải có {num_layers} phần tử, nhận {len(lambda_init)}"
            )
        self.register_buffer(
            "lambda_weights",
            torch.tensor(lambda_init, dtype=torch.float32),
            persistent=False,
        )
        self.use_dynamic = use_dynamic

    def forward(
        self,
        feats_a: List[torch.Tensor],
        feats_b: List[torch.Tensor],
        dynamic_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            feats_a: list đặc trưng decoder của Student A, length = num_layers.
            feats_b: tương tự cho Student B.
            dynamic_weights: tensor shape (num_layers,) - nếu dùng IMP-3.
        """
        if len(feats_a) != self.num_layers or len(feats_b) != self.num_layers:
            raise ValueError(
                f"Cần đúng {self.num_layers} feature maps, nhận "
                f"{len(feats_a)} và {len(feats_b)}"
            )

        # Chọn bộ trọng số
        if self.use_dynamic and dynamic_weights is not None:
            lam = dynamic_weights
        else:
            lam = self.lambda_weights.to(feats_a[0].device)

        total = feats_a[0].new_zeros(())
        for k, (fa, fb) in enumerate(zip(feats_a, feats_b)):
            kl = self._kl_aligned(fa, fb)
            total = total + lam[k] * kl
        return total

    @staticmethod
    def _kl_aligned(
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính KL divergence đối xứng giữa hai feature map.

        Vì feature chưa chuẩn hóa, ta chuẩn hóa về phân phối xác suất
        bằng softmax theo channel, rồi tính:
            KL(A||B) = Σ A * log(A / B)
        Loss = 0.5 * (KL(A||B) + KL(B||A)).
        """
        p = F.softmax(feat_a, dim=1)
        q = F.softmax(feat_b, dim=1)
        # Tránh log(0)
        eps = 1e-8
        kl_ab = (p * (p.add(eps).log() - q.add(eps).log())).sum(dim=1).mean()
        kl_ba = (q * (q.add(eps).log() - p.add(eps).log())).sum(dim=1).mean()
        return 0.5 * (kl_ab + kl_ba)


# ====================================================================== #
# 4. CMD LOSS (Eq. 5)                                                    #
# ====================================================================== #
class CMDLoss(nn.Module):
    """
    Cross Model Discrepancy Loss.

    Teacher (ảnh gốc) → pseudo-label.
    Student (ảnh che) → học bám pseudo-label của Teacher tương ứng.

    L_dis = (1/(N+M)) * Σ [ CE(p_A^m, ŷ_A^T) + CE(p_B^m, ŷ_B^T) ]
    """

    def __init__(self, ignore_index: int = -1) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        pred_a_masked: torch.Tensor,
        pred_b_masked: torch.Tensor,
        pseudo_a_teacher: torch.Tensor,
        pseudo_b_teacher: torch.Tensor,
    ) -> torch.Tensor:
        loss_a = self.ce(pred_a_masked, pseudo_a_teacher)
        loss_b = self.ce(pred_b_masked, pseudo_b_teacher)
        return loss_a + loss_b


# ====================================================================== #
# 5. GAUSSIAN RAMP-UP cho β                                              #
# ====================================================================== #
def gaussian_rampup(current_epoch: int, rampup_length: int = 50) -> float:
    """
    Hệ số β tăng dần theo hàm Gaussian.

    β(t) = exp( -5 * (1 - t/T)^2 ) với t = current_epoch, T = rampup_length.
    """
    if rampup_length <= 0:
        return 1.0
    if current_epoch >= rampup_length:
        return 1.0
    return float(math.exp(-5.0 * (1.0 - current_epoch / rampup_length) ** 2))


# ====================================================================== #
# 6. DYNAMIC λ_k (IMP-3)                                                #
# ====================================================================== #
class DynamicLambdaController:
    """
    Bộ điều khiển trọng số λ_k động cho CFC (Cải tiến IMP-3).

    Ý tưởng: tầng decoder nào có loss KL cao (hai nhánh đang "lệch" nhiều)
    thì tăng trọng số để ép hai nhánh học đồng bộ hơn. Ngược lại, tầng đã ổn
    định sẽ có trọng số giảm (tránh over-regularize).

    λ_k(t+1) = λ_k_init * ( EMA_loss_k(t) / mean(EMA_loss) )^gamma

    - EMA_loss_k là trung bình trượt lũy thừa của loss KL tại tầng k.
    - gamma điều khiển mức "tăng/giảm" (mặc định 1.0).
    """

    def __init__(
        self,
        num_layers: int = 4,
        lambda_init: Sequence[float] | None = None,
        ema_decay: float = 0.9,
        gamma: float = 1.0,
        min_lambda: float = 0.05,
        max_lambda: float = 2.0,
    ) -> None:
        self.num_layers = num_layers
        if lambda_init is None:
            lambda_init = [0.2 * (k + 1) for k in range(num_layers)]
        self.lambda_init = list(lambda_init)
        self.ema_decay = ema_decay
        self.gamma = gamma
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.register_ema()

    def register_ema(self) -> None:
        self.ema_losses = [0.0] * self.num_layers

    def step(self, layer_losses: List[float]) -> torch.Tensor:
        """
        Cập nhật EMA và trả về λ_k động.

        Args:
            layer_losses: list các loss KL thực tế tại mỗi tầng, length=num_layers.
        """
        assert len(layer_losses) == self.num_layers
        new_lambda = []
        for k, lk in enumerate(layer_losses):
            self.ema_losses[k] = (
                self.ema_decay * self.ema_losses[k]
                + (1 - self.ema_decay) * float(lk)
            )

        mean_ema = sum(self.ema_losses) / max(self.num_layers, 1) + 1e-8
        for k in range(self.num_layers):
            ratio = self.ema_losses[k] / mean_ema
            lam = self.lambda_init[k] * (ratio ** self.gamma)
            lam = max(self.min_lambda, min(self.max_lambda, lam))
            new_lambda.append(lam)

        return torch.tensor(new_lambda, dtype=torch.float32)


# ====================================================================== #
# 7. UNCERTAINTY MAP (IMP-2)                                             #
# ====================================================================== #
@torch.no_grad()
def compute_uncertainty_map(
    prob_a: torch.Tensor,
    prob_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Tính bản đồ bất định (uncertainty) từ hai dự đoán Teacher.

    Hai cách trộn:
    1. Disagreement: 1 - agreement(pred_A, pred_B) theo voxel.
    2. Predictive entropy của trung bình hai xác suất.

    Trả về bản đồ [0, 1] shape (B, 1, D, H, W):
        0 = rất tự tin/đồng thuận (dễ, thường là vùng nền),
        1 = rất bất định/không đồng thuận (ranh giới, cơ quan nhỏ).

    Args:
        prob_a: softmax probabilities từ Teacher A, (B, C, D, H, W).
        prob_b: softmax probabilities từ Teacher B, (B, C, D, H, W).
    """
    # 1. Disagreement: 1 nếu hai argmax khác nhau
    argmax_a = prob_a.argmax(dim=1, keepdim=True)
    argmax_b = prob_b.argmax(dim=1, keepdim=True)
    disagree = (argmax_a != argmax_b).float()

    # 2. Entropy trung bình
    mean_prob = 0.5 * (prob_a + prob_b)
    entropy = -(mean_prob * torch.log(mean_prob + eps)).sum(dim=1, keepdim=True)
    # Chuẩn hóa entropy về [0, 1]
    max_entropy = float(torch.log(torch.tensor(prob_a.shape[1], dtype=torch.float32)))
    entropy_norm = entropy / max_entropy

    # Kết hợp: ưu tiên disagreement (0/1) nhưng làm mượt bằng entropy
    uncertainty = torch.maximum(disagree, entropy_norm)

    # Chuẩn hóa về [0, 1] trong từng ảnh (batch-wise)
    B = uncertainty.shape[0]
    flat = uncertainty.view(B, -1)
    mn = flat.min(dim=1, keepdim=True)[0]
    mx = flat.max(dim=1, keepdim=True)[0]
    uncertainty = (flat - mn) / (mx - mn + eps)
    uncertainty = uncertainty.view_as(uncertainty)

    return uncertainty