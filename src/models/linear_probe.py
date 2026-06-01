"""
Non-Invasive Physical Probe (PhyIP) — strictly linear volatility forecaster.

Per the project plan's experimental protocol, the backbone is entirely frozen
and only this linear layer is trained. No non-linear activations, no deep MLP,
no fine-tuning of the physics encoder. This constraint ensures that any
predictive accuracy derives from the latent physical representations
themselves, not from the capacity of the downstream model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from loguru import logger


class LinearVolatilityProbe(nn.Module):
    """
    Strictly linear probe: hat_RV = W @ Z + b

    Can be fit either via:
    (a) closed-form ridge regression (fast, exact, preferred for static embeddings)
    (b) gradient descent (useful when embeddings are computed on-the-fly)
    """

    def __init__(self, embedding_dim: int = 768, n_horizons: int = 1, alpha: float = 1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_horizons = n_horizons
        self.alpha = alpha

        # Single linear layer, no bias term for strict linearity test,
        # bias added separately for numerical stability
        self.linear = nn.Linear(embedding_dim, n_horizons, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) embedding tensor from frozen backbone.
        Returns:
            rv_hat: (B, n_horizons) predicted realized volatility.
        """
        return self.linear(z)

    def fit_ridge(self, embeddings: np.ndarray, targets: np.ndarray):
        """
        Fit the linear probe via ridge regression (closed-form solution).

        This is the primary fitting method — faster and more numerically
        stable than gradient descent for a linear layer.

        Args:
            embeddings: (N, D) array of frozen backbone embeddings.
            targets: (N,) or (N, H) array of RV targets.
        """
        scaler = StandardScaler()
        Z = scaler.fit_transform(embeddings)

        ridge = Ridge(alpha=self.alpha, fit_intercept=True)
        ridge.fit(Z, targets)

        # Load fitted weights into the nn.Linear layer
        W = torch.tensor(ridge.coef_, dtype=torch.float32)
        b = torch.tensor(ridge.intercept_, dtype=torch.float32)

        if W.ndim == 1:
            W = W.unsqueeze(0)

        with torch.no_grad():
            self.linear.weight.copy_(W)
            self.linear.bias.copy_(b)

        self._scaler = scaler
        self._ridge = ridge

        logger.info(
            f"LinearProbe fitted: W shape={W.shape}, "
            f"alpha={self.alpha}, "
            f"train R^2={ridge.score(Z, targets):.4f}"
        )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict using fitted ridge regression."""
        if not hasattr(self, "_ridge"):
            raise RuntimeError("Call fit_ridge() before predict()")
        Z = self._scaler.transform(embeddings)
        return self._ridge.predict(Z)


class MultiHorizonLinearProbe(nn.Module):
    """
    Trains separate linear probes for each forecasting horizon.

    The project plan tests prediction at horizons: [10, 50, 100, 500] ticks.
    Each horizon gets its own ridge regression model fitted independently.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        horizons: Tuple[int, ...] = (10, 50, 100, 500),
        alpha: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.horizons = horizons
        self.alpha = alpha

        self.probes = nn.ModuleDict({
            str(h): LinearVolatilityProbe(embedding_dim, n_horizons=1, alpha=alpha)
            for h in horizons
        })

    def fit_all(self, embeddings: np.ndarray, rv_targets: Dict[int, np.ndarray]):
        """
        Fit one linear probe per horizon.

        Args:
            embeddings: (N, D) array.
            rv_targets: dict mapping horizon -> (N,) RV array.
        """
        for h in self.horizons:
            if h not in rv_targets:
                logger.warning(f"No targets for horizon {h}, skipping")
                continue
            targets = rv_targets[h]
            logger.info(f"Fitting probe for horizon={h}")
            self.probes[str(h)].fit_ridge(embeddings, targets)

    def predict_all(self, embeddings: np.ndarray) -> Dict[int, np.ndarray]:
        return {h: self.probes[str(h)].predict(embeddings) for h in self.horizons}

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {str(h): probe(z) for h, probe in self.probes.items()}


class PhysicsProbeModel(nn.Module):
    """
    Complete model: frozen physics backbone + linear probe.

    This is the full inference model for zero-shot latent transfer.
    The backbone processes LOB heatmaps; the probe predicts volatility.
    """

    def __init__(self, backbone: nn.Module, probe: MultiHorizonLinearProbe):
        super().__init__()
        self.backbone = backbone
        self.probe = probe

        # Enforce frozen backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            z = self.backbone(x)  # (B, D)
        return self.probe(z)

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x)
