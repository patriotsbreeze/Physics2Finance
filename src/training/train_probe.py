"""
Phase III: Zero-shot latent extraction and linear probe training.

Loads the frozen physics backbone, extracts embeddings from LOB heatmaps,
fits the strictly linear probe via ridge regression.

This is computationally inexpensive compared to pre-training — embedding
extraction is a forward pass through a frozen network, and ridge regression
has a closed-form solution.

Run with: python -m src.training.train_probe --config configs/probe_config.yaml
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from loguru import logger
import yaml

from src.models.vit_backbone import PhysicsViT
from src.models.linear_probe import MultiHorizonLinearProbe, PhysicsProbeModel
from src.data.financial.fi2010_loader import FI2010Dataset
from src.data.financial.lob_to_heatmap import LOBHeatmapEncoder


class EmbeddingExtractor:
    """
    Extracts frozen backbone embeddings from a DataLoader in batches.
    """

    def __init__(self, backbone: PhysicsViT, device: torch.device, batch_size: int = 256):
        self.backbone = backbone.to(device)
        self.backbone.eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def extract(self, dataset: Dataset) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        all_embeddings = []
        all_targets: Dict[int, list] = {}

        for batch in tqdm(loader, desc="Extracting embeddings"):
            images = batch["image"].to(self.device)
            targets = batch.get("targets", {})

            embeddings = self.backbone(images).cpu().numpy()  # (B, D)
            all_embeddings.append(embeddings)

            for h, rv in targets.items():
                if h not in all_targets:
                    all_targets[h] = []
                if torch.is_tensor(rv):
                    rv = rv.numpy()
                all_targets[h].append(rv)

        embeddings_arr = np.concatenate(all_embeddings, axis=0)
        targets_arr = {
            int(h): np.concatenate(v).astype(np.float32)
            for h, v in all_targets.items()
        }

        logger.info(
            f"Extracted embeddings: shape={embeddings_arr.shape}, "
            f"horizons={list(targets_arr.keys())}"
        )
        return embeddings_arr, targets_arr


def build_financial_datasets(cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    horizons = cfg["forecasting"]["horizons"]
    levels = cfg["data"]["lob_levels"]
    window = cfg["data"]["window_size"]
    img_size = cfg["data"]["img_size"]

    train_datasets, val_datasets, test_datasets = [], [], []

    for ds_cfg in cfg["data"]["datasets"]:
        name = ds_cfg["name"]
        path = ds_cfg["path"]

        if name == "fi2010":
            for split, container in [("train", train_datasets), ("val", val_datasets), ("test", test_datasets)]:
                ds = FI2010Dataset(
                    data_dir=path,
                    split=split,
                    window_size=window,
                    horizons=horizons,
                    lob_levels=levels,
                    img_size=img_size,
                )
                if len(ds) > 0:
                    container.append(ds)

    def concat_or_empty(dss):
        if not dss:
            logger.warning("Empty dataset split — check data paths")
            return None
        return ConcatDataset(dss) if len(dss) > 1 else dss[0]

    return concat_or_empty(train_datasets), concat_or_empty(val_datasets), concat_or_empty(test_datasets)


def main(config_path: str = "configs/probe_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["hardware"]["device"] if torch.cuda.is_available() else "cpu")

    # Load frozen backbone
    backbone = PhysicsViT(
        arch=cfg["model"]["backbone_arch"],
        frozen=cfg["model"]["freeze_backbone"],
    )
    ckpt_path = cfg["model"]["backbone_checkpoint"]
    if Path(ckpt_path).exists():
        backbone.load_physics_checkpoint(ckpt_path, key="teacher")
        logger.info(f"Loaded physics backbone from {ckpt_path}")
    else:
        logger.warning(
            f"Checkpoint not found: {ckpt_path}. "
            "Using randomly initialized backbone (ablation study mode)."
        )
    backbone.freeze()

    # Build financial datasets
    train_ds, val_ds, test_ds = build_financial_datasets(cfg)

    if train_ds is None:
        logger.error("No training data. Run scripts/download_data.sh and prepare financial data.")
        return

    # Extract embeddings from frozen backbone
    extractor = EmbeddingExtractor(backbone, device, cfg["hardware"]["batch_size"])

    logger.info("Extracting train embeddings...")
    train_emb, train_rv = extractor.extract(train_ds)

    logger.info("Extracting val embeddings...")
    val_emb, val_rv = extractor.extract(val_ds) if val_ds else (np.empty((0, backbone.embed_dim)), {})

    logger.info("Extracting test embeddings...")
    test_emb, test_rv = extractor.extract(test_ds) if test_ds else (np.empty((0, backbone.embed_dim)), {})

    # Save embeddings for reuse
    if cfg["output"]["save_embeddings"]:
        np.save(output_dir / "train_embeddings.npy", train_emb)
        np.save(output_dir / "test_embeddings.npy", test_emb)
        np.savez(output_dir / "train_rv_targets.npz", **{str(k): v for k, v in train_rv.items()})
        np.savez(output_dir / "test_rv_targets.npz", **{str(k): v for k, v in test_rv.items()})
        logger.info(f"Saved embeddings to {output_dir}")

    # Fit multi-horizon linear probe
    probe = MultiHorizonLinearProbe(
        embedding_dim=backbone.embed_dim,
        horizons=tuple(cfg["forecasting"]["horizons"]),
        alpha=cfg["probe"]["alpha"],
    )
    probe.fit_all(train_emb, train_rv)

    # Save probe
    torch.save(probe.state_dict(), output_dir / "linear_probe.pth")

    # Evaluate on test set
    if len(test_emb) > 0:
        from src.evaluation.metrics import evaluate_all_horizons
        test_preds = probe.predict_all(test_emb)
        results = evaluate_all_horizons(test_preds, test_rv, cfg["forecasting"]["horizons"])
        logger.info("Test results:")
        for h, metrics in results.items():
            logger.info(f"  Horizon {h}: {metrics}")

        np.save(output_dir / "test_predictions.npy", test_preds)

    logger.info("Linear probe training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe_config.yaml")
    args = parser.parse_args()
    main(args.config)
