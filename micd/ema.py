"""
Trình cập nhật trọng số EMA cho các mô hình Teacher.

EMA update (Eq. 1):
    θ'_t = α * θ'_{t-1} + (1 - α) * θ_t

Trong đó:
- θ'_t : trọng số Teacher tại bước t.
- θ_t  : trọng số Student tại bước t.
- α    : siêu tham số momentum (mặc định 0.99).
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class EMAUpdater:
    """
    Cập nhật trọng số Teacher = EMA của trọng số Student.

    Có hỗ trợ "warm-up": trong vài iteration đầu, copy thẳng từ Student
    sang Teacher thay vì EMA (giúp Teacher khởi đầu đúng hướng).
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        alpha: float = 0.99,
        warmup_steps: int = 0,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Đảm bảo Teacher bắt đầu là bản sao của Student
        self._copy_from_student()

    def _copy_from_student(self) -> None:
        """Copy toàn bộ tham số từ Student sang Teacher."""
        for t_param, s_param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            t_param.data.copy_(s_param.data)
        # Copy cả buffers (như running_mean/var của BN)
        for t_buf, s_buf in zip(self.teacher.buffers(), self.student.buffers()):
            t_buf.data.copy_(s_buf.data)

    @torch.no_grad()
    def update(self) -> None:
        """Cập nhật trọng số Teacher một bước."""
        self.step_count += 1
        if self.warmup_steps > 0 and self.step_count <= self.warmup_steps:
            self._copy_from_student()
            return

        for t_param, s_param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            t_param.data.mul_(self.alpha).add_(s_param.data, alpha=1.0 - self.alpha)

        # Buffers (BN stats) cũng EMA
        for t_buf, s_buf in zip(self.teacher.buffers(), self.student.buffers()):
            t_buf.data.mul_(self.alpha).add_(s_buf.data, alpha=1.0 - self.alpha)


def build_teacher_from_student(student: nn.Module) -> nn.Module:
    """Tạo một bản sao Teacher từ Student (cùng kiến trúc, trọng số giống hệt)."""
    teacher = deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher