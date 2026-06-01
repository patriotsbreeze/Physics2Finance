"""
Integration tests for the LOB → heatmap pipeline.
"""

import numpy as np
import pytest
from src.data.financial.lob_to_heatmap import (
    LOBHeatmapEncoder,
    compute_realized_volatility,
    build_lob_dataset_from_snapshots,
)


class TestBuildLOBDataset:
    def test_dataset_shape(self):
        rng = np.random.RandomState(0)
        n_ticks = 300
        levels = 10
        snapshots = rng.rand(n_ticks, 4 * levels).astype(np.float32)
        log_returns = rng.normal(0, 0.01, n_ticks).astype(np.float32)

        heatmaps, targets = build_lob_dataset_from_snapshots(
            snapshots, log_returns,
            window_size=50,
            horizons=(10, 20),
            levels=levels,
            img_size=64,
            stride=10,
        )

        assert heatmaps.ndim == 4
        assert heatmaps.shape[1] == 3
        assert heatmaps.shape[2] == 64
        assert 10 in targets and 20 in targets
        assert len(targets[10]) == len(heatmaps)

    def test_imbalance_channel_range(self):
        rng = np.random.RandomState(1)
        window = rng.rand(50, 40).astype(np.float32)
        enc = LOBHeatmapEncoder(levels=10, img_size=32)
        hm = enc.encode(window)
        # All channels in [0, 1] after normalization
        assert hm[2].min() >= 0.0
        assert hm[2].max() <= 1.0 + 1e-5
