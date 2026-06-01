"""
DINO self-supervised loss for fluid dynamics pre-training.

Implements the knowledge distillation loss from Caron et al. (2021).
A student network processes both local and global crops of a fluid
snapshot; a momentum-updated teacher network processes only global crops.
The student is trained to predict the teacher's softmax distribution,
forcing it to learn multi-scale, shift-invariant physical representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List
import numpy as np
from loguru import logger


class DINOLoss(nn.Module):
    """
    DINO cross-entropy loss between student and teacher softmax outputs.

    Teacher outputs are centered (to prevent collapse) and sharpened
    with a lower temperature. Student processes all crops (local+global);
    teacher processes only global crops.
    """

    def __init__(
        self,
        out_dim: int = 65536,
        n_crops: int = 10,  # 2 global + 8 local
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        n_epochs: int = 100,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = n_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Teacher temperature schedule (warmup from low to target)
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(n_epochs - warmup_teacher_temp_epochs) * teacher_temp,
        ])

    def forward(
        self,
        student_output: List[torch.Tensor],
        teacher_output: List[torch.Tensor],
        epoch: int,
    ) -> torch.Tensor:
        """
        Args:
            student_output: list of (B, out_dim) logits from all crops.
            teacher_output: list of (B, out_dim) logits from global crops only.
            epoch: current training epoch (used for temperature schedule).

        Returns:
            Scalar cross-entropy loss.
        """
        student_out = torch.stack(student_output) / self.student_temp  # (n_crops, B, D)
        student_out = student_out.chunk(self.n_crops)

        teacher_temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax(
            (torch.stack(teacher_output) - self.center) / teacher_temp, dim=-1
        )
        teacher_out = teacher_out.chunk(2)  # only 2 global crops

        total_loss = 0.0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue  # skip self-prediction
                loss = -torch.sum(q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self._update_center(teacher_out)
        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_output: tuple):
        """EMA update of the centering vector to prevent mode collapse."""
        batch_center = torch.cat(teacher_output, dim=0).mean(dim=0, keepdim=True)

        # Distributed reduce if running multi-GPU
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= dist.get_world_size()

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MomentumUpdater:
    """
    Manages exponential moving average (EMA) updates for the teacher network.
    Teacher weights: theta_t = m * theta_t + (1-m) * theta_s
    """

    def __init__(self, base_momentum: float = 0.996, final_momentum: float = 1.0):
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum

    def get_momentum(self, step: int, total_steps: int) -> float:
        """Cosine schedule from base_momentum to final_momentum."""
        return self.final_momentum - (self.final_momentum - self.base_momentum) * (
            np.cos(np.pi * step / total_steps) + 1
        ) / 2

    @torch.no_grad()
    def update(self, student: nn.Module, teacher: nn.Module, momentum: float):
        for p_s, p_t in zip(student.parameters(), teacher.parameters()):
            p_t.data.mul_(momentum).add_((1 - momentum) * p_s.data)
