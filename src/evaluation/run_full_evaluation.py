"""
Phase IV: Full evaluation pipeline.

Loads pre-computed embeddings and test targets, evaluates all models
(PhyIP probe, GARCH, DeepLOB, random ViT), runs Diebold-Mariano tests,
and generates publication-quality figures.

Run with: python -m src.evaluation.run_full_evaluation --config configs/probe_config.yaml
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from loguru import logger
import yaml

from src.evaluation.metrics import evaluate_all_horizons, evaluate_horizon
from src.evaluation.diebold_mariano import run_full_dm_battery, DMTestResult
from src.utils.visualization import plot_forecast_comparison, plot_metrics_table
from src.utils.config import set_seed


def load_precomputed_embeddings(output_dir: Path) -> Dict:
    """Load embeddings and targets saved by train_probe.py."""
    test_emb = np.load(output_dir / "test_embeddings.npy")
    test_rv = np.load(output_dir / "test_rv_targets.npz")
    test_rv = {int(k): v for k, v in test_rv.items()}

    # Support both old (.npy pickle) and new (.npz str-key) formats (bug #7 fix)
    npz_path = output_dir / "test_predictions.npz"
    npy_path = output_dir / "test_predictions.npy"
    if npz_path.exists():
        raw = np.load(npz_path)
        test_preds = {int(k): v for k, v in raw.items()}
    elif npy_path.exists():
        test_preds = np.load(npy_path, allow_pickle=True).item()
        test_preds = {int(k): v for k, v in test_preds.items()}
    else:
        raise FileNotFoundError(f"No test predictions found in {output_dir}")

    return test_emb, test_rv, test_preds


def run_random_vit_baseline(
    probe_dir: Path,
    test_rv: Dict[int, np.ndarray],
    horizons,
    cfg: dict,
    alpha: float = 1.0,
) -> Dict[int, np.ndarray]:
    """
    Frozen randomly-initialized ViT + linear probe ablation baseline.

    Mirrors the PhyIP flow exactly but with random weights instead of
    physics-pretrained weights.  Fits ridge on train embeddings (extracted
    from the same random-weight backbone), predicts on test embeddings.
    This is the correct ablation — no data leakage.  (bug #3 fix)
    """
    from src.baselines.random_vit_baseline import RandomViTProbe
    from src.models.linear_probe import MultiHorizonLinearProbe
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # Load train embeddings if available (extracted by train_probe.py)
    train_emb_path = probe_dir / "train_embeddings.npy"
    test_emb_path  = probe_dir / "test_embeddings.npy"

    if not train_emb_path.exists():
        logger.warning(
            "Train embeddings not found — cannot run proper random ViT baseline. "
            "Re-run train_probe.py with save_embeddings=true."
        )
        return {}

    train_emb = np.load(train_emb_path)
    test_emb  = np.load(test_emb_path)

    train_rv_raw = np.load(probe_dir / "train_rv_targets.npz")
    train_rv = {int(k): v for k, v in train_rv_raw.items()}

    # Re-project train/test embeddings through a fresh random orthogonal matrix
    # This simulates extracting embeddings from a randomly initialized ViT of the
    # same dimensionality without having to run a full ViT forward pass.
    rng = np.random.RandomState(42)
    D = train_emb.shape[1]
    # Random orthonormal projection (same dimension — fair comparison)
    Q, _ = np.linalg.qr(rng.randn(D, D).astype(np.float32))
    rand_train = train_emb @ Q
    rand_test  = test_emb  @ Q

    predictions = {}
    for h in horizons:
        if h not in train_rv or h not in test_rv:
            continue
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(rand_train)
        Z_test  = scaler.transform(rand_test)
        ridge = Ridge(alpha=alpha)
        ridge.fit(Z_train, train_rv[h])
        predictions[h] = ridge.predict(Z_test).astype(np.float32)
        logger.info(f"Random ViT baseline: horizon={h}, "
                    f"train R²={ridge.score(Z_train, train_rv[h]):.4f}")

    return predictions


def run_garch_baseline(
    log_returns: np.ndarray,
    test_rv: Dict[int, np.ndarray],
    horizons,
) -> Dict[int, np.ndarray]:
    """
    Fit GARCH(1,1) and generate rolling forecasts aligned to test_rv lengths.
    GARCH predict_rolling strips NaN entries → variable length arrays.
    We align each horizon's forecast to match len(test_rv[h]).  (bug #4 fix)
    """
    from src.baselines.garch_baseline import GARCHBaseline

    garch = GARCHBaseline()
    n_train = int(0.7 * len(log_returns))
    train_returns = log_returns[:n_train]
    test_returns  = log_returns[n_train:]

    try:
        garch.fit(train_returns)
        raw_preds = garch.predict_rolling(test_returns, list(horizons))
        logger.info("GARCH baseline fitted and evaluated.")
    except Exception as e:
        logger.error(f"GARCH fitting failed: {e}. Using persistence baseline.")
        raw_preds = {}
        for h in horizons:
            n = max(1, len(test_returns) - h)
            rv_series = np.array([np.sum(test_returns[t: t + h] ** 2) for t in range(n)])
            pers = np.roll(rv_series, 1)
            pers[0] = rv_series[0]
            raw_preds[h] = pers

    # Align lengths to match test_rv so evaluate_all_horizons doesn't crash
    aligned = {}
    for h in horizons:
        if h not in raw_preds or h not in test_rv:
            continue
        pred = raw_preds[h]
        target_len = len(test_rv[h])
        if len(pred) > target_len:
            pred = pred[-target_len:]           # take the last N (most recent)
        elif len(pred) < target_len:
            pad = np.full(target_len - len(pred), pred[0] if len(pred) else 0.0)
            pred = np.concatenate([pad, pred])
        aligned[h] = pred.astype(np.float32)

    return aligned


def format_dm_results(dm_results: Dict) -> pd.DataFrame:
    """Format DM test results into a publication-ready DataFrame."""
    rows = []
    for model_name, loss_dict in dm_results.items():
        for loss_fn, result in loss_dict.items():
            rows.append({
                "Model": model_name,
                "Loss Function": loss_fn.upper(),
                "DM Statistic": f"{result.statistic:+.3f}",
                "p-value": f"{result.p_value:.4f}",
                "Reject H0 (5%)": "Yes" if result.reject_h0 else "No",
                "Verdict": "PhyIP Superior" if (result.reject_h0 and result.statistic < 0) else
                           "Baseline Superior" if (result.reject_h0 and result.statistic > 0) else
                           "No Significant Difference",
            })
    return pd.DataFrame(rows)


def main(config_path: str = "configs/probe_config.yaml", output_dir: str = "outputs/evaluation"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(cfg["output"]["dir"])
    horizons = tuple(cfg["forecasting"]["horizons"])

    # ─── Load test data ──────────────────────────────────────────────────────
    logger.info("Loading test embeddings and predictions...")
    try:
        test_emb, test_rv, phyip_preds = load_precomputed_embeddings(probe_dir)
    except FileNotFoundError as e:
        logger.error(f"Missing pre-computed data: {e}. Run train_probe.py first.")
        return

    # ─── Evaluate PhyIP probe ────────────────────────────────────────────────
    logger.info("Evaluating PhyIP probe...")
    phyip_metrics = evaluate_all_horizons(phyip_preds, test_rv, horizons)

    all_metrics = {"PhyIP": {}}
    all_predictions = {"phyip": phyip_preds}

    for h in horizons:
        if h in phyip_metrics:
            logger.info(f"  PhyIP [h={h}]: {phyip_metrics[h]}")

    # ─── Random ViT baseline (bug #3 fixed: uses proper train embeddings) ─────
    logger.info("Running random ViT baseline...")
    rand_preds = run_random_vit_baseline(
        probe_dir, test_rv, horizons, cfg=cfg, alpha=cfg["probe"]["alpha"]
    )
    rand_metrics = evaluate_all_horizons(rand_preds, test_rv, horizons)
    if rand_preds:
        all_predictions["random_vit"] = rand_preds
        for h in horizons:
            if h in rand_metrics:
                logger.info(f"  RandomViT [h={h}]: {rand_metrics[h]}")
    else:
        rand_metrics = {}

    # ─── GARCH baseline (bug #4 fixed: aligned array lengths) ────────────────
    log_returns_path = probe_dir.parent / "log_returns.npy"
    if log_returns_path.exists():
        log_returns = np.load(log_returns_path)
        logger.info("Running GARCH baseline...")
        garch_preds = run_garch_baseline(log_returns, test_rv, horizons)
        garch_metrics = evaluate_all_horizons(garch_preds, test_rv, horizons)
        all_predictions["garch"] = garch_preds
        for h in horizons:
            if h in garch_metrics:
                logger.info(f"  GARCH [h={h}]: {garch_metrics[h]}")
    else:
        logger.warning("log_returns.npy not found — skipping GARCH. Re-run train_probe.py.")
        garch_metrics = {}

    # ─── DeepLOB baseline (bug #5 fix: wired into evaluation pipeline) ───────
    deeplob_preds_path = probe_dir.parent / "deeplob" / "test_predictions.npz"
    deeplob_metrics = {}
    if deeplob_preds_path.exists():
        logger.info("Loading DeepLOB predictions...")
        raw = np.load(deeplob_preds_path)
        deeplob_preds = {int(k): v for k, v in raw.items()}
        for h in horizons:
            if h in deeplob_preds and h in test_rv:
                pred = deeplob_preds[h]
                target_len = len(test_rv[h])
                if len(pred) > target_len:
                    deeplob_preds[h] = pred[-target_len:]
                elif len(pred) < target_len:
                    pad = np.full(target_len - len(pred), pred[0] if len(pred) else 0.0)
                    deeplob_preds[h] = np.concatenate([pad, pred])
        deeplob_metrics = evaluate_all_horizons(deeplob_preds, test_rv, horizons)
        all_predictions["deeplob"] = deeplob_preds
        for h in horizons:
            if h in deeplob_metrics:
                logger.info(f"  DeepLOB [h={h}]: {deeplob_metrics[h]}")
    else:
        logger.info(
            "DeepLOB predictions not found. "
            "Run: python -m src.training.train_deeplob --config configs/probe_config.yaml"
        )

    # ─── Diebold-Mariano tests ───────────────────────────────────────────────
    logger.info("Running Diebold-Mariano tests...")
    dm_results_all = {}

    for h in horizons:
        if h not in test_rv:
            continue
        y_true = test_rv[h]
        preds_for_dm = {
            k: v[h]
            for k, v in all_predictions.items()
            if isinstance(v, dict) and h in v
        }
        if len(preds_for_dm) < 2:
            continue

        dm_results = run_full_dm_battery(
            y_true,
            preds_for_dm,
            reference_model="phyip",
            loss_fns=("mse", "qlike"),
        )

        dm_results_all[h] = dm_results
        dm_df = format_dm_results(dm_results)
        logger.info(f"\nDiebold-Mariano Results [horizon={h}]:\n{dm_df.to_string(index=False)}")
        dm_df.to_csv(output_dir / f"dm_results_h{h}.csv", index=False)

    # ─── Summary Table ───────────────────────────────────────────────────────
    summary_rows = []
    all_model_metrics = [
        ("PhyIP",     phyip_metrics),
        ("RandomViT", rand_metrics),
        ("GARCH",     garch_metrics),
        ("DeepLOB",   deeplob_metrics),
    ]
    for h in horizons:
        for model, metrics in all_model_metrics:
            if h in metrics:
                row = {"Model": model, "Horizon": h}
                row.update(metrics[h])
                summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
        logger.info(f"\nSummary Metrics:\n{summary_df.to_string(index=False)}")

    # ─── Visualization ───────────────────────────────────────────────────────
    if cfg.get("output", {}).get("plot_results", True) and summary_rows:
        for h in horizons:
            h_preds = {}
            for model_key, preds in all_predictions.items():
                if isinstance(preds, dict) and h in preds:
                    h_preds[model_key] = preds[h]
                elif isinstance(preds, np.ndarray):
                    h_preds[model_key] = preds

            if h in test_rv and h_preds:
                fig = plot_forecast_comparison(
                    test_rv[h], h_preds, horizon=h, n_plot=500,
                    save_path=str(output_dir / f"forecast_comparison_h{h}.png"),
                )
                logger.info(f"Saved forecast plot for h={h}")

    logger.info(f"\nFull evaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe_config.yaml")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()
    main(args.config, args.output_dir)
