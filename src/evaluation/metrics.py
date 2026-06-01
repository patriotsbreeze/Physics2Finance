"""
Evaluation metrics for volatility forecasting.

Implements MAE, RMSE, and QLIKE loss functions per the project plan.
QLIKE is the standard loss function for volatility model evaluation,
robust to microstructure noise and penalizing under-predictions more
severely than over-predictions.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. Penalizes large errors heavily."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    QLIKE (quasi-likelihood) loss for volatility forecasting.

    QLIKE = mean( RV/RV_hat - log(RV/RV_hat) - 1 )

    Derived from the log-normal quasi-likelihood. Strictly convex,
    asymmetric: under-prediction (RV_hat < RV) penalized more than
    over-prediction. Robust to microstructure noise in RV estimates.
    """
    ratio = np.maximum(y_true, eps) / np.maximum(y_pred, eps)
    return float(np.mean(ratio - np.log(ratio) - 1))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def evaluate_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute all metrics for a single forecasting horizon."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    assert len(y_true) == len(y_pred), f"Shape mismatch: {len(y_true)} vs {len(y_pred)}"

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "qlike": qlike(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "n": int(len(y_true)),
    }


def evaluate_all_horizons(
    predictions: Dict[int, np.ndarray],
    targets: Dict[int, np.ndarray],
    horizons: Tuple[int, ...],
) -> Dict[int, Dict[str, float]]:
    """Evaluate metrics for all forecasting horizons."""
    results = {}
    for h in horizons:
        if h not in predictions or h not in targets:
            continue
        results[h] = evaluate_horizon(targets[h], predictions[h])
    return results


def loss_differential(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    loss_fn: str = "mse",
) -> np.ndarray:
    """
    Compute the pointwise loss differential d_t = L(y, y1_hat) - L(y, y2_hat).
    Used as input to the Diebold-Mariano test.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred1 = np.asarray(y_pred1, dtype=np.float64)
    y_pred2 = np.asarray(y_pred2, dtype=np.float64)

    if loss_fn == "mse":
        l1 = (y_true - y_pred1) ** 2
        l2 = (y_true - y_pred2) ** 2
    elif loss_fn == "qlike":
        eps = 1e-8
        r1 = np.maximum(y_true, eps) / np.maximum(y_pred1, eps)
        r2 = np.maximum(y_true, eps) / np.maximum(y_pred2, eps)
        l1 = r1 - np.log(r1) - 1
        l2 = r2 - np.log(r2) - 1
    elif loss_fn == "mae":
        l1 = np.abs(y_true - y_pred1)
        l2 = np.abs(y_true - y_pred2)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return l1 - l2
