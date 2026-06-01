"""
LOB-to-heatmap encoder: converts raw LOB snapshot sequences to RGB images.

Channel mapping (per project plan physics analogy):
  R: Ask depth (pressure/resistance to upward price movement)
  G: Bid depth (fluid density/viscous drag)
  B: Order imbalance (velocity field/local advection)

Heatmap layout:
  X-axis: time steps (oldest → newest)
  Y-axis: price levels relative to mid (ask levels top, bid levels bottom)
  Pixel intensity: normalized volume at that price × time coordinate
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


class LOBHeatmapEncoder:
    """
    Converts a sequence of LOB snapshots into a (3, img_size, img_size) RGB tensor.

    Expected LOB matrix format (FI-2010 / DeepLOB convention):
      Columns interleaved per level: [P_ask_1, V_ask_1, P_bid_1, V_bid_1,
                                       P_ask_2, V_ask_2, P_bid_2, V_bid_2, ...]
      Shape: (window_size, 4 * lob_levels)
    """

    def __init__(
        self,
        lob_levels: int = 10,
        img_size: int = 224,
        normalize: bool = True,
    ):
        self.lob_levels = lob_levels
        self.img_size = img_size
        self.normalize = normalize

    def encode_matrix(self, lob_matrix: np.ndarray) -> torch.Tensor:
        """
        Args:
            lob_matrix: (T, 4*k) array — interleaved [P_ask_i, V_ask_i, P_bid_i, V_bid_i]

        Returns:
            (3, img_size, img_size) float32 tensor
        """
        k = self.lob_levels
        # Extract price/volume columns from interleaved format
        ask_prices = lob_matrix[:, 0::4][:, :k]   # (T, k)
        ask_vols   = lob_matrix[:, 1::4][:, :k]   # (T, k)
        bid_prices = lob_matrix[:, 2::4][:, :k]   # (T, k)
        bid_vols   = lob_matrix[:, 3::4][:, :k]   # (T, k)
        return self.encode(ask_prices, ask_vols, bid_prices, bid_vols)

    def encode(
        self,
        ask_prices: np.ndarray,
        ask_vols: np.ndarray,
        bid_prices: np.ndarray,
        bid_vols: np.ndarray,
    ) -> torch.Tensor:
        """
        Args:
            ask_prices, ask_vols, bid_prices, bid_vols: each (T, k)

        Returns:
            (3, img_size, img_size) float32 tensor
        """
        T, k = ask_vols.shape
        eps = 1e-8

        # Normalize volumes relative to total depth per snapshot
        total = ask_vols.sum(1, keepdims=True) + bid_vols.sum(1, keepdims=True) + eps
        ask_norm = ask_vols / total   # (T, k)
        bid_norm = bid_vols / total   # (T, k)

        # Order imbalance ∈ [-1, 1] → shift to [0, 1]
        imb = (bid_vols.sum(1) - ask_vols.sum(1)) / (
            bid_vols.sum(1) + ask_vols.sum(1) + eps
        )
        imb_01 = (imb + 1.0) / 2.0   # (T,)

        # Build spatial grid: rows = 2k price levels, cols = T time steps
        # Row layout: row 0..k-1 = ask_level_k..ask_level_1 (worst→best ask)
        #             row k..2k-1 = bid_level_1..bid_level_k (best→worst bid)
        H = 2 * k

        # R channel: ask depth
        R = np.zeros((H, T), dtype=np.float32)
        R[:k, :] = ask_norm[:, ::-1].T     # flip so best ask is at center (row k-1)

        # G channel: bid depth
        G = np.zeros((H, T), dtype=np.float32)
        G[k:, :] = bid_norm.T              # best bid at row k

        # B channel: order imbalance broadcast over price dimension
        B = np.tile(imb_01[np.newaxis, :].astype(np.float32), (H, 1))

        heatmap = np.stack([R, G, B], axis=0)   # (3, H, T)

        # Resize to (3, img_size, img_size)
        t = torch.from_numpy(heatmap).unsqueeze(0)   # (1, 3, H, T)
        t = F.interpolate(t, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False).squeeze(0)

        if self.normalize:
            for c in range(3):
                ch = t[c]
                mn, mx = ch.min(), ch.max()
                if mx > mn:
                    t[c] = (ch - mn) / (mx - mn)

        return t   # (3, img_size, img_size)

    def mid_prices_from_matrix(self, lob_matrix: np.ndarray) -> np.ndarray:
        """Return mid-price series from raw LOB matrix."""
        best_ask = lob_matrix[:, 0]   # col 0 = best ask price
        best_bid = lob_matrix[:, 2]   # col 2 = best bid price
        return (best_ask + best_bid) / 2.0
