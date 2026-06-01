"""
Diebold-Mariano (DM) test for forecast accuracy comparison.

Tests the null hypothesis H0: E[d_t] = 0 (equal forecast accuracy)
against H1: E[d_t] ≠ 0 (one forecast is superior).

For multi-step volatility forecasts, serial correlation in d_t is
expected — we use Newey-West (HAC) standard errors per the project plan.
The DM statistic is asymptotically N(0,1) under H0.

Reference: Diebold & Mariano (1995), "Comparing Predictive Accuracy",
           Journal of Business & Economic Statistics.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class DMTestResult:
    statistic: float
    p_value: float
    loss_differential_mean: float
    newey_west_std: float
    n_obs: int
    lag: int
    loss_fn: str
    reject_h0: bool  # at 5% significance

    def __str__(self) -> str:
        direction = "Model 1 superior" if self.statistic < 0 else "Model 2 superior"
        verdict = f"REJECT H0 ({direction})" if self.reject_h0 else "FAIL TO REJECT H0 (equal accuracy)"
        return (
            f"Diebold-Mariano Test [{self.loss_fn}]\n"
            f"  DM statistic = {self.statistic:+.4f}\n"
            f"  p-value      = {self.p_value:.4f}\n"
            f"  E[d_t]       = {self.loss_differential_mean:.6f}\n"
            f"  NW std       = {self.newey_west_std:.6f}\n"
            f"  n_obs        = {self.n_obs} (lag={self.lag})\n"
            f"  Verdict: {verdict}"
        )


def newey_west_variance(
    d: np.ndarray,
    lag: int,
) -> float:
    """
    Compute the Newey-West HAC variance estimate of sqrt(T) * d_bar.

    NW_var = gamma_0 + 2 * sum_{j=1}^{lag} (1 - j/(lag+1)) * gamma_j

    where gamma_j = T^{-1} * sum_{t=j+1}^{T} (d_t - d_bar)(d_{t-j} - d_bar)

    This is the standard HAC correction for multi-step ahead forecasts
    which induces moving-average dependence in the loss differential.
    """
    T = len(d)
    d_centered = d - d.mean()

    # Autocovariances
    gamma = np.zeros(lag + 1)
    for j in range(lag + 1):
        gamma[j] = np.mean(d_centered[j:] * d_centered[:T - j]) if j < T else 0.0

    # Bartlett weights
    nw_var = gamma[0]
    for j in range(1, lag + 1):
        weight = 1.0 - j / (lag + 1)
        nw_var += 2.0 * weight * gamma[j]

    return max(nw_var, 1e-30)


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    loss_fn: str = "mse",
    lag: Optional[int] = None,
    alternative: str = "two-sided",
) -> DMTestResult:
    """
    Conduct the Diebold-Mariano test.

    H0: Models 1 and 2 have equal predictive accuracy.
    H1 (two-sided): accuracy differs.
    H1 (less): Model 1 is more accurate (d_t < 0 on average).
    H1 (greater): Model 2 is more accurate (d_t > 0 on average).

    Args:
        y_true: true realized volatility values, shape (T,).
        y_pred1: predictions from Model 1 (e.g., PhysIP probe), shape (T,).
        y_pred2: predictions from Model 2 (e.g., GARCH), shape (T,).
        loss_fn: "mse", "qlike", or "mae".
        lag: Newey-West lag. If None, uses floor(T^{1/3}) as rule-of-thumb.
        alternative: "two-sided", "less", or "greater".

    Returns:
        DMTestResult with test statistic, p-value, and verdict.
    """
    from src.evaluation.metrics import loss_differential as compute_ld

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred1 = np.asarray(y_pred1, dtype=np.float64).ravel()
    y_pred2 = np.asarray(y_pred2, dtype=np.float64).ravel()

    T = len(y_true)
    assert len(y_pred1) == T and len(y_pred2) == T, "All arrays must have equal length"

    d = compute_ld(y_true, y_pred1, y_pred2, loss_fn=loss_fn)
    d_bar = d.mean()

    if lag is None:
        lag = max(1, int(np.floor(T ** (1 / 3))))

    nw_var = newey_west_variance(d, lag)
    nw_std = np.sqrt(nw_var / T)

    if nw_std < 1e-15:
        dm_stat = 0.0
        p_value = 1.0
    else:
        dm_stat = float(d_bar / nw_std)

        if alternative == "two-sided":
            p_value = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
        elif alternative == "less":
            p_value = float(stats.norm.cdf(dm_stat))
        elif alternative == "greater":
            p_value = float(1 - stats.norm.cdf(dm_stat))
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

    return DMTestResult(
        statistic=dm_stat,
        p_value=p_value,
        loss_differential_mean=float(d_bar),
        newey_west_std=float(nw_std),
        n_obs=T,
        lag=lag,
        loss_fn=loss_fn,
        reject_h0=(p_value < 0.05),
    )


def run_full_dm_battery(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    reference_model: str = "phyip",
    loss_fns: Tuple[str, ...] = ("mse", "qlike", "mae"),
) -> Dict[str, Dict[str, DMTestResult]]:
    """
    Run DM tests comparing reference model against all other models,
    across all loss functions.

    Returns nested dict: {model_name: {loss_fn: DMTestResult}}
    """
    results = {}
    ref_preds = predictions[reference_model]

    for model_name, preds in predictions.items():
        if model_name == reference_model:
            continue
        results[model_name] = {}
        for loss_fn in loss_fns:
            result = diebold_mariano_test(
                y_true, ref_preds, preds, loss_fn=loss_fn
            )
            results[model_name][loss_fn] = result

    return results
