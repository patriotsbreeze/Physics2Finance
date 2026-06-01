"""
Randomly initialized frozen ViT + linear probe ablation baseline.

This is the critical ablation study from the project plan: if a randomly
initialized (untrained) ViT can predict volatility almost as well as the
physics-pre-trained one, then the predictive power comes from the
high-dimensional random projection capacity of the architecture, not
from the learned Navier-Stokes representations.

Rejection of this null (physics ViT >> random ViT) is required to
confirm the cross-domain physical homology claim.
"""

import torch
import numpy as np
from typing import Dict, Tuple
from loguru import logger

from src.models.vit_backbone import PhysicsViT
from src.models.linear_probe import MultiHorizonLinearProbe


class RandomViTProbe:
    """
    Frozen randomly initialized ViT + ridge regression.
    Exact same pipeline as PhyIP but with random weights instead of
    physics-pre-trained weights.
    """

    def __init__(
        self,
        arch: str = "vit_base_patch16_224",
        img_size: int = 224,
        embedding_dim: int = 768,
        horizons: Tuple[int, ...] = (10, 50, 100, 500),
        alpha: float = 1.0,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.backbone = PhysicsViT(arch=arch, img_size=img_size, frozen=True)
        self.probe = MultiHorizonLinearProbe(
            embedding_dim=embedding_dim,
            horizons=horizons,
            alpha=alpha,
        )
        logger.info(f"RandomViT: frozen random weights (seed={seed}), arch={arch}")

    def fit(
        self,
        train_images: torch.Tensor,
        train_rv: Dict[int, np.ndarray],
        device: torch.device,
        batch_size: int = 256,
    ):
        """Extract embeddings from random ViT and fit linear probe."""
        embeddings = self._extract_embeddings(train_images, device, batch_size)
        self.probe.fit_all(embeddings, train_rv)

    def predict(
        self,
        images: torch.Tensor,
        device: torch.device,
        batch_size: int = 256,
    ) -> Dict[int, np.ndarray]:
        embeddings = self._extract_embeddings(images, device, batch_size)
        return self.probe.predict_all(embeddings)

    @torch.no_grad()
    def _extract_embeddings(
        self,
        images: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> np.ndarray:
        self.backbone.to(device).eval()
        all_emb = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            emb = self.backbone(batch).cpu().numpy()
            all_emb.append(emb)
        return np.concatenate(all_emb, axis=0)
