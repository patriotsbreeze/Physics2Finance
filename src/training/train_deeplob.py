"""
DeepLOB supervised training script.

Trains the DeepLOB CNN-LSTM baseline on the same FI-2010 / Binance data used
for the PhyIP linear probe, producing a fully supervised competitive baseline.

DeepLOBDataset produces raw LOB matrices in (B, 1, 2k, T) format expected by
DeepLOBTrainer.  This is separate from the heatmap format used by PhyIP.

Run with:
  python -m src.training.train_deeplob --config configs/probe_config.yaml

Output: outputs/deeplob/deeplob_best.pth
        outputs/deeplob/test_predictions.npz  (used by run_full_evaluation.py)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from loguru import logger
import yaml

from src.baselines.deeplob_baseline import DeepLOB, DeepLOBTrainer
from src.data.financial.fi2010_loader import FI2010Dataset
from src.data.financial.binance_loader import BinanceDataset
from src.evaluation.metrics import evaluate_all_horizons


class DeepLOBDataset(Dataset):
    """
    Wraps FI2010Dataset / BinanceDataset to produce raw LOB matrices in the
    (1, 2k, window_size) format expected by DeepLOB, plus stacked RV targets.

    DeepLOB receives the raw volume matrix — NOT the RGB heatmap.  It learns
    its own spatial-temporal features end-to-end.

    Input format to DeepLOB:
      batch["lob"]:  (B, 1, 2k, window_size) float32
      batch["rv"]:   (B, n_horizons) float32
    """

    def __init__(
        self,
        source_dataset: Dataset,
        horizons: List[int],
        lob_levels: int = 10,
    ):
        self.source = source_dataset
        self.horizons = horizons
        self.lob_levels = lob_levels

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> dict:
        sample = self.source[idx]
        targets = sample.get("targets", {})

        # Reconstruct raw LOB matrix from source dataset internal state
        # The source dataset has _lob_matrix and _indices attributes
        src = self.source
        # Handle ConcatDataset wrapping
        if hasattr(src, "datasets"):
            # Find the actual dataset for this idx
            cumulative = 0
            for ds in src.datasets:
                if idx < cumulative + len(ds):
                    return self._get_from_dataset(ds, idx - cumulative, targets)
                cumulative += len(ds)
            return self._fallback(sample, targets)
        else:
            return self._get_from_dataset(src, idx, targets)

    def _get_from_dataset(self, ds: Dataset, idx: int, targets: dict) -> dict:
        if not hasattr(ds, "_lob_matrix") or not hasattr(ds, "_indices"):
            return self._fallback(None, targets)

        t = ds._indices[idx]
        window = ds._lob_matrix[t - ds.window_size: t]   # (T, 4k)

        k = self.lob_levels
        # Extract volume columns from interleaved format [P_a,V_a,P_b,V_b,...]
        ask_vols = window[:, 1::4][:, :k]   # (T, k)
        bid_vols = window[:, 3::4][:, :k]   # (T, k)

        # Stack bid and ask volumes: (2k, T)
        lob_vol = np.concatenate([ask_vols.T, bid_vols.T], axis=0)   # (2k, T)
        lob_tensor = torch.from_numpy(lob_vol.astype(np.float32)).unsqueeze(0)  # (1, 2k, T)

        # Normalize volume per time step
        vol_sum = lob_tensor.sum(dim=1, keepdim=True).clamp(min=1e-8)
        lob_tensor = lob_tensor / vol_sum

        # Stack RV targets for all horizons
        rv_list = [targets.get(h, torch.tensor(0.0, dtype=torch.float32)) for h in self.horizons]
        rv_tensor = torch.stack(rv_list)   # (n_horizons,)

        return {"lob": lob_tensor, "rv": rv_tensor}

    @staticmethod
    def _fallback(sample, targets: dict) -> dict:
        # Cannot reconstruct raw LOB — return zeros
        return {
            "lob": torch.zeros(1, 20, 100),
            "rv":  torch.zeros(4),
        }


def build_deeplob_datasets(cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    horizons = cfg["forecasting"]["horizons"]
    levels   = cfg["data"]["lob_levels"]
    window   = cfg["data"]["window_size"]

    train_hm, val_hm, test_hm = [], [], []

    for ds_cfg in cfg["data"]["datasets"]:
        name = ds_cfg["name"]
        path = ds_cfg["path"]

        if name == "fi2010":
            for split, container in [("train", train_hm), ("val", val_hm), ("test", test_hm)]:
                ds = FI2010Dataset(
                    data_dir=path, split=split,
                    window_size=window, horizons=horizons, lob_levels=levels,
                )
                if len(ds) > 0:
                    container.append(DeepLOBDataset(ds, horizons, levels))

        elif name == "binance_btcusdt":
            for split, container in [("train", train_hm), ("val", val_hm), ("test", test_hm)]:
                ds = BinanceDataset(
                    data_path=path, split=split,
                    window_size=window, horizons=horizons, lob_levels=levels,
                )
                if len(ds) > 0:
                    container.append(DeepLOBDataset(ds, horizons, levels))

    def _combine(lst):
        if not lst:
            return None
        return ConcatDataset(lst) if len(lst) > 1 else lst[0]

    return _combine(train_hm), _combine(val_hm), _combine(test_hm)


def main(config_path: str = "configs/probe_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"]["dir"]).parent / "deeplob"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["hardware"]["device"] if torch.cuda.is_available() else "cpu")

    horizons = cfg["forecasting"]["horizons"]
    levels   = cfg["data"]["lob_levels"]
    window   = cfg["data"]["window_size"]

    logger.info("Building DeepLOB datasets...")
    train_ds, val_ds, test_ds = build_deeplob_datasets(cfg)

    if train_ds is None:
        logger.error("No training data. Download and prepare financial data first.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=cfg["hardware"]["batch_size"],
        shuffle=True, num_workers=cfg["hardware"]["num_workers"],
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["hardware"]["batch_size"],
        shuffle=False, num_workers=0,
    ) if val_ds else None
    test_loader = DataLoader(
        test_ds, batch_size=cfg["hardware"]["batch_size"],
        shuffle=False, num_workers=0,
    ) if test_ds else None

    model = DeepLOB(
        lob_levels=levels,
        window_size=window,
        n_horizons=len(horizons),
    )
    logger.info(f"DeepLOB parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = DeepLOBTrainer(model=model, device=device)

    # Training loop
    n_epochs = cfg.get("deeplob", {}).get("epochs", 50)
    patience = cfg.get("deeplob", {}).get("patience", 10)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        train_loss = trainer.train_epoch(train_loader)

        if val_loader:
            val_metrics = trainer.evaluate(val_loader)
            val_loss = val_metrics.get("mse", float("inf"))
            trainer.scheduler.step(val_loss)
            logger.info(
                f"Epoch {epoch:3d}: train_loss={train_loss:.6f} "
                f"val_mse={val_metrics.get('mse', float('nan')):.6f} "
                f"val_r2={val_metrics.get('r2', float('nan')):.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "cfg": cfg,
                }
                torch.save(ckpt, output_dir / "deeplob_best.pth")
                logger.info(f"  Saved best checkpoint (val_mse={val_loss:.6f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        else:
            logger.info(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}")
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       output_dir / "deeplob_best.pth")

    # Load best model and evaluate on test set
    ckpt_path = output_dir / "deeplob_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')}")

    if test_loader:
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["lob"].to(device)
                y = batch["rv"]
                pred = model(x).cpu().numpy()
                all_preds.append(pred)
                all_targets.append(y.numpy())

        preds_arr   = np.concatenate(all_preds,   axis=0)   # (N, n_horizons)
        targets_arr = np.concatenate(all_targets, axis=0)   # (N, n_horizons)

        # Save per-horizon predictions as npz with str keys
        pred_dict = {str(h): preds_arr[:, i] for i, h in enumerate(horizons)}
        np.savez(output_dir / "test_predictions.npz", **pred_dict)

        # Evaluate per horizon
        preds_by_h   = {h: preds_arr[:, i].astype(np.float32) for i, h in enumerate(horizons)}
        targets_by_h = {h: targets_arr[:, i].astype(np.float32) for i, h in enumerate(horizons)}
        metrics = evaluate_all_horizons(preds_by_h, targets_by_h, tuple(horizons))
        for h, m in metrics.items():
            logger.info(f"  DeepLOB test [h={h}]: {m}")

    logger.info(f"DeepLOB training complete. Outputs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe_config.yaml")
    args = parser.parse_args()
    main(args.config)
