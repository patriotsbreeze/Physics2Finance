"""
FI-2010 Limit Order Book Dataset loader.

FI-2010 contains 10-level LOB snapshots for Nokia, WRT, and Kesko stocks
on NASDAQ Helsinki.  Each row is one event-driven snapshot with 144 features
(40 raw LOB + 104 hand-crafted) and 5 classification labels.

We discard the original classification labels and compute our own realized
volatility targets for the horizons specified in probe_config.yaml.

Data files follow the pattern:
  Train_Dst_NoAuction_DecPre_CF_7.txt   (training)
  Test_Dst_NoAuction_DecPre_CF_7.txt    (testing)

Download from: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
Place .txt files in: data/financial/fi2010/
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger

from src.data.financial.lob_to_heatmap import LOBHeatmapEncoder


class FI2010Dataset(Dataset):
    """
    Sliding-window dataset over FI-2010 LOB snapshots.

    Each sample is a dict:
      "image":   (3, img_size, img_size) float32 tensor — LOB heatmap
      "targets": {horizon: float32 scalar} — realized volatility at each horizon
      "mid_price": float32 — mid-price at the end of the window

    Column layout of raw FI-2010 files (first 40 of 144 features):
      Interleaved per level: [P_ask_i, V_ask_i, P_bid_i, V_bid_i] for i=1..10
      Total: 40 columns.  Best ask = col 0, best bid = col 2.
    """

    # Standard FI-2010 train/val/test day split
    # The dataset covers 10 trading days; we use 7/2/1
    _SPLIT_DAYS = {
        "train": list(range(1, 8)),    # days 1–7
        "val":   list(range(8, 10)),   # days 8–9
        "test":  [10],                 # day 10
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        window_size: int = 100,
        horizons: List[int] = (10, 50, 100, 500),
        lob_levels: int = 10,
        img_size: int = 224,
        stride: int = 1,
        normalize_prices: bool = True,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.data_dir = Path(data_dir)
        self.split = split
        self.window_size = window_size
        self.horizons = list(horizons)
        self.lob_levels = lob_levels
        self.img_size = img_size
        self.stride = stride
        self.normalize_prices = normalize_prices

        self.encoder = LOBHeatmapEncoder(lob_levels=lob_levels, img_size=img_size)

        self._data: Optional[np.ndarray] = None   # (N, 40) raw LOB features
        self._mid_prices: Optional[np.ndarray] = None
        self._log_returns: Optional[np.ndarray] = None
        self._indices: List[int] = []

        self._load()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self):
        raw_arrays = []

        txt_files = sorted(
            glob.glob(str(self.data_dir / "*.txt"))
            + glob.glob(str(self.data_dir / "**" / "*.txt"))
        )

        if not txt_files:
            logger.warning(
                f"No FI-2010 .txt files found in {self.data_dir}. "
                "Download from https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649"
            )
            return

        # Filter files by split prefix (Train_* vs Test_*)
        if self.split in ("train", "val"):
            txt_files = [f for f in txt_files if "Train" in os.path.basename(f)]
        else:
            txt_files = [f for f in txt_files if "Test" in os.path.basename(f)]

        if not txt_files:
            logger.warning(f"No matching files for split='{self.split}' in {self.data_dir}")
            return

        for fpath in txt_files:
            try:
                arr = np.loadtxt(fpath)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                # FI-2010 files are stored (n_features, n_samples) — transpose to (n_samples, n_features)
                if arr.shape[0] < arr.shape[1]:
                    arr = arr.T
                raw_arrays.append(arr)
                logger.info(f"Loaded {fpath}: shape={arr.shape} (samples, features)")
            except Exception as e:
                logger.error(f"Failed to load {fpath}: {e}")

        if not raw_arrays:
            return

        full = np.concatenate(raw_arrays, axis=0).astype(np.float32)
        n_total = len(full)

        # Use first 40 columns (raw LOB features)
        lob_cols = min(4 * self.lob_levels, full.shape[1])
        self._data = full[:, :lob_cols]

        # For train/val splits, use proportional time slices since we
        # typically only have Train files and no per-day file labeling
        if self.split == "train":
            end = int(0.7 * n_total)
            self._data = self._data[:end]
        elif self.split == "val":
            s = int(0.7 * n_total)
            e = int(0.8 * n_total)
            self._data = self._data[s:e]
        # test: use all of the test file data

        # Compute mid-prices and log-returns
        self._mid_prices = self.encoder.mid_prices_from_matrix(self._data)
        mp = self._mid_prices.copy()
        mp = np.where(mp <= 0, np.nan, mp)
        self._log_returns = np.concatenate([[0.0], np.diff(np.log(mp))])
        self._log_returns = np.nan_to_num(self._log_returns, nan=0.0)

        # Build valid window indices: need window_size + max_horizon points ahead
        max_h = max(self.horizons) if self.horizons else 0
        n = len(self._data)
        self._indices = list(range(
            self.window_size,
            n - max_h,
            self.stride,
        ))

        logger.info(
            f"FI2010Dataset [{self.split}]: {len(self._indices)} windows "
            f"from {n} snapshots (window={self.window_size}, horizons={self.horizons})"
        )

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        t = self._indices[idx]
        window = self._data[t - self.window_size: t]   # (window_size, 40)

        if self.normalize_prices:
            window = self._normalize_window(window)

        image = self.encoder.encode_matrix(window)

        targets = {}
        for h in self.horizons:
            if t + h <= len(self._log_returns):
                rv = float(np.sum(self._log_returns[t: t + h] ** 2))
                targets[h] = torch.tensor(rv, dtype=torch.float32)

        return {
            "image":     image,
            "targets":   targets,
            "mid_price": float(self._mid_prices[t]),
        }

    def get_log_returns(self) -> np.ndarray:
        """Return full log-return series (used to save log_returns.npy for GARCH)."""
        if self._log_returns is None:
            return np.array([])
        return self._log_returns.copy()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_window(window: np.ndarray) -> np.ndarray:
        """
        Normalize prices to percentage deviations from the last mid-price.
        Volumes normalized by total volume in the window.
        Keeps the LOB structure visually comparable across stocks/days.
        """
        w = window.copy()
        # Price columns: 0, 2, 4, 6, ... (every other starting from 0)
        price_cols = list(range(0, w.shape[1], 2))
        vol_cols   = list(range(1, w.shape[1], 2))

        mid = (w[-1, 0] + w[-1, 2]) / 2.0
        if mid > 0:
            w[:, price_cols] = (w[:, price_cols] - mid) / mid

        max_vol = np.abs(w[:, vol_cols]).max()
        if max_vol > 0:
            w[:, vol_cols] = w[:, vol_cols] / max_vol

        return w
