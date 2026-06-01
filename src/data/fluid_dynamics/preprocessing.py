"""
Multi-scale patch extraction (DINO-style augmentation) for fluid field pre-training.

Implements the two-view self-distillation strategy from Caron et al. (2021):
  - 2 global crops: large random resized crops (scale 0.4–1.0)
  - N local crops: small crops (scale 0.05–0.4) for multi-scale context

For fluid fields (unlike natural images), color jitter is replaced by
physics-appropriate augmentations:
  - Spatial flips (both axes are valid due to periodic boundary conditions)
  - 90° rotations (isotropic turbulence is rotationally invariant)
  - Gaussian noise (mimics numerical solver noise)
  - Channel shuffle NOT applied — u/v/p ordering is semantically meaningful
"""

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class PhysicsAugment:
    """
    Physics-appropriate augmentation for fluid field snapshots.

    Unlike natural image augmentations, we avoid color jitter (which distorts
    physical field values) and instead use spatial symmetry operations valid
    for the isotropic turbulence training data.
    """

    def __init__(
        self,
        img_size: int = 224,
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        flip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        noise_std: float = 0.01,
    ):
        self.img_size = img_size
        self.crop_scale = crop_scale
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_std = noise_std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (3, H, W) float32 tensor

        Returns:
            (3, img_size, img_size) augmented tensor
        """
        _, H, W = img.shape

        # Random resized crop
        scale = random.uniform(*self.crop_scale)
        crop_h = max(1, int(H * scale))
        crop_w = max(1, int(W * scale))
        top    = random.randint(0, max(0, H - crop_h))
        left   = random.randint(0, max(0, W - crop_w))
        img = img[:, top:top + crop_h, left:left + crop_w]

        # Resize to target
        img = F.interpolate(
            img.unsqueeze(0), size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        # Spatial flips (valid for periodic boundary turbulence)
        if random.random() < self.flip_prob:
            img = torch.flip(img, dims=[2])   # horizontal flip
        if random.random() < self.flip_prob:
            img = torch.flip(img, dims=[1])   # vertical flip

        # 90° rotation (isotropic turbulence is rotationally symmetric)
        if random.random() < self.rotate_prob:
            k = random.randint(1, 3)
            img = torch.rot90(img, k=k, dims=[1, 2])

        # Additive Gaussian noise (mimics solver noise)
        if self.noise_std > 0:
            img = img + torch.randn_like(img) * self.noise_std
            img = img.clamp(0.0, 1.0)

        return img


class MultiScalePatchExtractor:
    """
    Generates multiple views of a fluid field at different scales.

    Returns: [global_crop_1, global_crop_2, local_crop_1, ..., local_crop_N]

    The first two crops are large (global context), the remaining N crops are
    small (local features).  This matches the DINO student/teacher setup where
    the teacher sees only global crops.
    """

    def __init__(
        self,
        img_size: int = 224,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale:  Tuple[float, float] = (0.05, 0.4),
        n_local_crops: int = 8,
        noise_std: float = 0.01,
    ):
        self.n_local_crops = n_local_crops

        self.global_transform = PhysicsAugment(
            img_size=img_size,
            crop_scale=global_crops_scale,
            flip_prob=0.5,
            rotate_prob=0.5,
            noise_std=noise_std,
        )
        self.local_transform = PhysicsAugment(
            img_size=img_size,
            crop_scale=local_crops_scale,
            flip_prob=0.5,
            rotate_prob=0.3,
            noise_std=noise_std * 0.5,
        )

    def __call__(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            img: (3, H, W) float32 tensor (single fluid snapshot)

        Returns:
            List of length 2 + n_local_crops, each (3, img_size, img_size)
        """
        crops = [
            self.global_transform(img),
            self.global_transform(img),
        ]
        crops += [self.local_transform(img) for _ in range(self.n_local_crops)]
        return crops


def compute_dataset_channel_stats(
    dataset: torch.utils.data.Dataset,
    n_samples: int = 500,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean and std over a random sample of the dataset.
    Used to initialize PhysicalFieldNorm statistics before pre-training.

    Returns:
        mean: (3,) tensor
        std:  (3,) tensor
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:min(n_samples, len(indices))]

    channel_sums   = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    total_pixels   = 0

    for i in indices:
        sample = dataset[i]
        img = sample["image"]   # (3, H, W)
        for c in range(img.shape[0]):
            channel_sums[c]   += img[c].sum().item()
            channel_sq_sum[c] += (img[c] ** 2).sum().item()
        total_pixels += img.shape[1] * img.shape[2]

    mean = channel_sums / total_pixels
    var  = channel_sq_sum / total_pixels - mean ** 2
    std  = torch.sqrt(var.clamp(min=1e-6))

    return mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
