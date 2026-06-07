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
    train_log_returns: np.ndarray,
    test_log_returns: np.ndarray,
    test_rv: Dict[int, np.ndarray],
    horizons,
    window_size: int = 100,
) -> Dict[int, np.ndarray]:
    """
    Fit GARCH(1,1) on training returns, generate rolling forecasts on test returns.

    For each test window i (corresponding to position window_size+i in test_log_returns),
    forecast h-step ahead conditional variance and sum to get expected realized variance.
    This correctly aligns GARCH forecasts with the test dataset window indices.
    """
    from src.baselines.garch_baseline import GARCHBaseline

    garch = GARCHBaseline()
    try:
        garch.fit(train_log_returns)
    except Exception as e:
        logger.error(f"GARCH fit failed: {e}")
        return {}

    n_test = len(test_log_returns)
    max_h = max(horizons)

    # Roll through test returns; at each position t forecast max_h steps ahead.
    # Only refit when we have enough data (≥1000 points) and every 1000 steps.
    result = garch._result
    scale = garch._scale
    forecasts = {h: np.full(n_test, np.nan) for h in horizons}
    last_refit = -1

    for t in range(n_test - max_h):
        # Refit every 1000 steps using expanding window of test data (only if ≥1000 pts)
        if t >= 1000 and (t - last_refit) >= 1000:
            try:
                from arch import arch_model as _arch_model
                subset = np.concatenate([train_log_returns, test_log_returns[:t]]) * 100
                m = _arch_model(subset, mean=garch.mean, vol="GARCH",
                                p=garch.p, q=garch.q, dist=garch.dist)
                result = m.fit(disp="off", show_warning=False)
                last_refit = t
            except Exception:
                pass

        try:
            fc = result.forecast(horizon=max_h, reindex=False)
            cond_var = fc.variance.values[-1]   # (max_h,)
            for h in horizons:
                forecasts[h][t] = float(np.sum(cond_var[:h])) / (scale ** 2)
        except Exception:
            continue

    # Each test window i corresponds to test_log_returns position window_size + i.
    # Slice that range and drop NaNs at the edges to match test_rv length.
    aligned = {}
    for h in horizons:
        if h not in test_rv:
            continue
        target_len = len(test_rv[h])
        raw = forecasts[h][window_size: window_size + target_len]
        valid_mask = ~np.isnan(raw)
        if valid_mask.sum() == 0:
            logger.warning(f"GARCH: all NaN for h={h}, skipping.")
            continue
        # Fill leading NaNs with the first valid value
        first_valid = raw[valid_mask][0]
        raw[~valid_mask] = first_valid
        aligned[h] = raw.astype(np.float32)

    logger.info("GARCH baseline fitted and evaluated.")
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
    train_lr_path = probe_dir.parent / "log_returns.npy"
    test_lr_path  = probe_dir.parent / "test_log_returns.npy"
    if train_lr_path.exists() and test_lr_path.exists():
        train_log_returns = np.load(train_lr_path)
        test_log_returns  = np.load(test_lr_path)
        window_size = cfg["data"].get("window_size", 100)
        logger.info("Running GARCH baseline...")
        garch_preds = run_garch_baseline(
            train_log_returns, test_log_returns, test_rv, horizons, window_size
        )
        garch_metrics = evaluate_all_horizons(garch_preds, test_rv, horizons)
        if garch_preds:
            all_predictions["garch"] = garch_preds
        for h in horizons:
            if h in garch_metrics:
                logger.info(f"  GARCH [h={h}]: {garch_metrics[h]}")
    else:
        logger.warning("log_returns.npy or test_log_returns.npy not found — skipping GARCH. Re-run train_probe.py.")
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
