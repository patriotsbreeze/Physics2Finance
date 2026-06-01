"""
GARCH(1,1) baseline for volatility forecasting.

The GARCH(1,1) model is the undisputed industry standard for
conditional variance estimation. It represents the null hypothesis
that historical price returns alone are sufficient for volatility
prediction — no microstructural data required.

sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}

The forecast horizon is extended by iterating the recursion forward.
Uses the arch library for robust MLE parameter estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch library not installed. GARCH baseline unavailable. Install with: pip install arch")


class GARCHBaseline:
    """
    GARCH(1,1) volatility forecaster.

    Fit on in-sample log-return series, then generates multi-step
    ahead conditional variance forecasts aligned with the same horizons
    used by the linear probe.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "StudentsT",
        mean: str = "Constant",
    ):
        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self._model = None
        self._result = None

    def fit(self, log_returns: np.ndarray, verbose: bool = False) -> "GARCHBaseline":
        """
        Fit GARCH(1,1) to the training log-return series.

        Args:
            log_returns: 1D array of log price returns (scaled by 100 for numerical stability).
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch library required: pip install arch")

        returns = np.asarray(log_returns, dtype=np.float64) * 100  # rescale

        model = arch_model(
            returns,
            mean=self.mean,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
        )
        self._result = model.fit(
            disp="off" if not verbose else "final",
            show_warning=False,
        )
        self._model = model
        self._scale = 100.0

        logger.info(
            f"GARCH({self.p},{self.q}) fitted: "
            f"omega={self._result.params.get('omega', 'N/A'):.4f}, "
            f"alpha={self._result.params.get('alpha[1]', 'N/A'):.4f}, "
            f"beta={self._result.params.get('beta[1]', 'N/A'):.4f}"
        )
        return self

    def predict_rolling(
        self,
        log_returns: np.ndarray,
        horizons: List[int],
        refit_every: int = 500,
    ) -> Dict[int, np.ndarray]:
        """
        Generate rolling out-of-sample volatility forecasts.

        Uses expanding window: at each test point t, refit on all
        available history (or use last-fit parameters) and forecast
        multi-step ahead conditional variance.

        Args:
            log_returns: full out-of-sample return series.
            horizons: list of forecast horizons (in ticks).
            refit_every: number of steps between model refits.

        Returns:
            Dict mapping horizon -> array of RV forecasts.
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch library required: pip install arch")

        n = len(log_returns)
        forecasts = {h: np.full(n, np.nan) for h in horizons}

        result = self._result  # initial fit
        last_refit = 0

        for t in range(0, n):
            if t - last_refit >= refit_every and t > 100:
                # Refit on expanding window
                try:
                    subset = log_returns[:t] * 100
                    model = arch_model(subset, mean=self.mean, vol="GARCH", p=self.p, q=self.q, dist=self.dist)
                    result = model.fit(disp="off", show_warning=False)
                    last_refit = t
                except Exception:
                    pass  # keep last valid fit

            if result is None:
                continue

            max_h = max(horizons)
            try:
                fc = result.forecast(horizon=max_h, reindex=False)
                cond_var = fc.variance.values[-1]  # (max_h,) array
            except Exception:
                continue

            for h in horizons:
                if t + h <= n:
                    # Sum of conditional variances = expected realized variance
                    rv_forecast = float(np.sum(cond_var[:h]) / (self._scale ** 2))
                    forecasts[h][t] = rv_forecast

        # Remove leading NaNs
        results = {}
        for h in horizons:
            valid = ~np.isnan(forecasts[h])
            results[h] = forecasts[h][valid]

        return results

    def forecast_next(self, log_returns: np.ndarray, horizon: int) -> float:
        """Single-shot forward forecast given a return history."""
        if self._result is None:
            raise RuntimeError("Call fit() before forecasting")
        fc = self._result.forecast(horizon=horizon, reindex=False)
        cond_var = fc.variance.values[-1]
        return float(np.sum(cond_var[:horizon]) / (self._scale ** 2))

    @property
    def params(self) -> dict:
        if self._result is None:
            return {}
        return dict(self._result.params)
