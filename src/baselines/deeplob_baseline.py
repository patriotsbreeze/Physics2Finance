"""
DeepLOB baseline: fully supervised CNN-LSTM for LOB-based volatility prediction.

Reference: Zhang et al. (2019), "DeepLOB: Deep Convolutional Neural Networks
           for Limit Order Books", IEEE Transactions on Signal Processing.

Architecture:
  1. Inception-style CNN: parallel 1×1, 1×2, 1×4 convolutions for spatial
     feature extraction from the LOB depth matrix.
  2. 1D CNN over price-level axis for order book shape features.
  3. LSTM: captures temporal dependencies in LOB dynamics.
  4. FC head: predicts realized volatility (regression mode) or
     mid-price movement direction (original classification mode).

This model is trained end-to-end (invasively) on financial data,
unlike the frozen PhyIP probe. Serves as the "best possible supervised
baseline" — the upper bound for performance achievable with labeled data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from loguru import logger


class InceptionModule(nn.Module):
    """
    Inception-style multi-scale feature extraction block.
    Captures spatial patterns at multiple receptive field sizes simultaneously.
    """

    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        branch_out = out_channels // 4

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out, kernel_size=1),
            nn.BatchNorm2d(branch_out),
            nn.LeakyReLU(0.01),
        )
        self.branch1x2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out, kernel_size=(1, 2), padding=(0, 0)),
            nn.BatchNorm2d(branch_out),
            nn.LeakyReLU(0.01),
        )
        self.branch1x4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out, kernel_size=(1, 4), padding=(0, 1)),
            nn.BatchNorm2d(branch_out),
            nn.LeakyReLU(0.01),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(in_channels, branch_out, kernel_size=1),
            nn.BatchNorm2d(branch_out),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1x1(x)

        b2 = self.branch1x2(x)
        # Pad to match spatial dims
        b2 = F.pad(b2, (0, x.shape[3] - b2.shape[3]))

        b3 = self.branch1x4(x)
        if b3.shape[3] != x.shape[3]:
            b3 = F.pad(b3, (0, x.shape[3] - b3.shape[3]))

        b4 = self.branch_pool(x)

        # Match time dims before concat
        min_t = min(b1.shape[3], b2.shape[3], b3.shape[3], b4.shape[3])
        return torch.cat([b[:, :, :, :min_t] for b in [b1, b2, b3, b4]], dim=1)


class DeepLOB(nn.Module):
    """
    DeepLOB: CNN-LSTM for fully supervised LOB volatility forecasting.

    Input: (B, 1, 2k, T) LOB matrix — 2k price levels, T time steps.
    Output: (B, n_horizons) realized volatility predictions.
    """

    def __init__(
        self,
        lob_levels: int = 10,
        window_size: int = 100,
        n_horizons: int = 4,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lob_levels = lob_levels
        self.window_size = window_size

        # Spatial CNN: extract order book shape features
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.01),
        )

        # Inception module: multi-scale temporal patterns
        self.inception = InceptionModule(cnn_channels, cnn_channels)

        # Pool over price-level dimension
        self.price_pool = nn.AdaptiveAvgPool2d((1, None))

        # LSTM: temporal dynamics
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Regression head for volatility prediction
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, n_horizons),
            nn.Softplus(),  # RV is strictly positive
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 2k, T) LOB volume matrix.
        Returns:
            rv_pred: (B, n_horizons) volatility predictions.
        """
        B = x.shape[0]

        # Spatial features: (B, C, H', T)
        x = self.spatial_cnn(x)

        # Inception: (B, C, H', T)
        x = self.inception(x)

        # Pool over price-level: (B, C, 1, T) -> (B, C, T)
        x = self.price_pool(x).squeeze(2)

        # Reshape for LSTM: (B, T, C)
        x = x.permute(0, 2, 1)

        # LSTM: (B, T, lstm_hidden)
        x, _ = self.lstm(x)

        # Use last timestep
        x = x[:, -1]

        return self.head(x)


class DeepLOBTrainer:
    """Training wrapper for DeepLOB with standard supervised training loop."""

    def __init__(
        self,
        model: DeepLOB,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )

    def train_epoch(self, loader, loss_fn=F.mse_loss) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            x = batch["lob"].to(self.device)
            y = batch["rv"].to(self.device)

            pred = self.model(x)
            loss = loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        from src.evaluation.metrics import evaluate_horizon
        self.model.eval()
        all_preds, all_targets = [], []

        for batch in loader:
            x = batch["lob"].to(self.device)
            y = batch["rv"].cpu().numpy()
            pred = self.model(x).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y)

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        return evaluate_horizon(targets.mean(axis=1), preds.mean(axis=1))
