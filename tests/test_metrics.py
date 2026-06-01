"""
Unit tests for evaluation metrics and Diebold-Mariano test.
"""

import numpy as np
import pytest
from src.evaluation.metrics import mae, rmse, qlike, loss_differential, r_squared
from src.evaluation.diebold_mariano import diebold_mariano_test, newey_west_variance


class TestMetrics:
    def test_mae_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_rmse_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_qlike_minimum_at_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        # QLIKE = 0 when y_hat = y (ratio=1, log ratio=0)
        assert qlike(y, y) == pytest.approx(0.0, abs=1e-8)

    def test_qlike_positive(self):
        y = np.array([1.0, 2.0, 3.0])
        y_hat = np.array([0.5, 1.5, 4.0])
        assert qlike(y, y_hat) > 0

    def test_qlike_asymmetric(self):
        # Under-prediction penalized more than over-prediction
        y = np.array([1.0])
        under = qlike(y, np.array([0.5]))   # under-predict by 50%
        over = qlike(y, np.array([2.0]))    # over-predict by 100%
        assert under > over

    def test_r_squared_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r_squared(y, y) == pytest.approx(1.0)

    def test_r_squared_null(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_mean = np.full_like(y, y.mean())
        assert r_squared(y, y_mean) == pytest.approx(0.0, abs=1e-8)

    def test_loss_differential_mse(self):
        y = np.array([1.0, 2.0, 3.0])
        p1 = np.array([1.1, 2.1, 3.1])  # smaller errors
        p2 = np.array([2.0, 3.0, 4.0])  # larger errors
        d = loss_differential(y, p1, p2, loss_fn="mse")
        assert d.mean() < 0  # Model 1 has lower loss


class TestDieboldMariano:
    def test_identical_forecasts(self):
        rng = np.random.RandomState(0)
        y = rng.exponential(0.1, 200)
        p = rng.exponential(0.1, 200)
        result = diebold_mariano_test(y, p, p)
        assert result.statistic == pytest.approx(0.0, abs=1e-10)
        assert result.p_value == pytest.approx(1.0, abs=1e-8)
        assert not result.reject_h0

    def test_clearly_superior_model(self):
        rng = np.random.RandomState(1)
        y = rng.exponential(0.1, 1000)
        p_good = y + rng.normal(0, 0.001, 1000)  # tiny errors
        p_bad = rng.exponential(0.1, 1000)         # random forecasts
        result = diebold_mariano_test(y, p_good, p_bad, loss_fn="mse")
        # Model 1 (p_good) should be clearly superior
        assert result.reject_h0
        assert result.statistic < 0

    def test_newey_west_nonnegative(self):
        rng = np.random.RandomState(2)
        d = rng.normal(0, 1, 100)
        nw_var = newey_west_variance(d, lag=5)
        assert nw_var >= 0

    def test_qlike_loss_fn(self):
        rng = np.random.RandomState(3)
        y = rng.exponential(0.1, 200) + 0.001
        p1 = y * rng.uniform(0.9, 1.1, 200)
        p2 = y * rng.uniform(0.5, 1.5, 200)  # wider errors
        result = diebold_mariano_test(y, p1, p2, loss_fn="qlike")
        # p1 has smaller QLIKE loss, so DM stat should be negative
        assert result.statistic < 0


class TestLOBHeatmap:
    def test_output_shape(self):
        from src.data.financial.lob_to_heatmap import LOBHeatmapEncoder
        enc = LOBHeatmapEncoder(levels=10, img_size=224)
        window = np.random.rand(100, 40).astype(np.float32)
        heatmap = enc.encode(window)
        assert heatmap.shape == (3, 224, 224)

    def test_output_range(self):
        from src.data.financial.lob_to_heatmap import LOBHeatmapEncoder
        enc = LOBHeatmapEncoder(levels=10, img_size=224)
        window = np.random.rand(100, 40).astype(np.float32)
        heatmap = enc.encode(window)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-6

    def test_realized_volatility(self):
        from src.data.financial.lob_to_heatmap import compute_realized_volatility
        log_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        rv = compute_realized_volatility(log_returns, horizon=3)
        assert len(rv) == len(log_returns) - 3
        assert rv[0] == pytest.approx(0.01**2 + (-0.02)**2 + 0.03**2, rel=1e-5)


try:
    import torch as _torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestLinearProbe:
    def test_fit_and_predict(self):
        from src.models.linear_probe import LinearVolatilityProbe
        rng = np.random.RandomState(42)
        D, N = 64, 200
        Z = rng.randn(N, D).astype(np.float32)
        W_true = rng.randn(D).astype(np.float32)
        y = Z @ W_true + rng.normal(0, 0.01, N)

        probe = LinearVolatilityProbe(embedding_dim=D)
        probe.fit_ridge(Z, y)
        y_hat = probe.predict(Z)
        assert y_hat.shape == (N,)

        corr = np.corrcoef(y, y_hat)[0, 1]
        assert corr > 0.99

    def test_frozen_backbone_no_grad(self):
        import torch
        from src.models.vit_backbone import PhysicsViT
        vit = PhysicsViT(arch="vit_tiny_patch16_224", frozen=True)
        for p in vit.parameters():
            assert not p.requires_grad
