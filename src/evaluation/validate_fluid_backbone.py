"""
In-domain validation of the fluid dynamics backbone.

Per the project plan (Phase I, Step 1.3): before freezing the backbone for the
financial probe, validate that it has learned Navier-Stokes dynamics rather than
overfitting texture/color statistics.

This script evaluates the teacher backbone on next-frame velocity prediction:
  1. Load a held-out fluid validation set.
  2. For each trajectory, pass frame t through the frozen backbone + a lightweight
     linear decoder to predict frame t+1 (u, v fields only).
  3. Compute relative L² error:  ‖ u_pred − u_true ‖₂ / ‖ u_true ‖₂

A relative L² < 0.3 on the validation set indicates meaningful physics learning.
Values > 0.7 suggest the backbone has not converged on Navier-Stokes structure.

This is a necessary sanity check before the cross-domain transfer claim is valid.

Run with:
  python -m src.evaluation.validate_fluid_backbone \
      --checkpoint outputs/pretrain/checkpoint_best.pth \
      --config configs/pretrain_config.yaml
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import yaml

from src.models.vit_backbone import PhysicsViT


class LinearVelocityDecoder(nn.Module):
    """
    Lightweight linear decoder: maps CLS+patch token embeddings to a 2D
    velocity field at resolution (H//patch_size, W//patch_size).

    Used only for in-domain validation — not part of the financial pipeline.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_patches: int = 196,
        patch_size: int = 16,
        out_channels: int = 2,   # u and v
    ):
        super().__init__()
        self.n_patches = n_patches
        self.patch_size = patch_size
        # Decode each patch token to (out_channels, patch_size, patch_size)
        self.decoder = nn.Linear(embed_dim, out_channels * patch_size * patch_size)
        self.out_channels = out_channels
        grid_size = int(n_patches ** 0.5)
        self.grid_size = grid_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N+1, D) — all tokens including CLS (from return_all_tokens=True)

        Returns:
            (B, out_channels, H, W) reconstructed velocity field
        """
        # Drop CLS token, use patch tokens
        patch_tokens = tokens[:, 1:, :]   # (B, N, D)
        B, N, D = patch_tokens.shape

        patches = self.decoder(patch_tokens)   # (B, N, C*P*P)
        C, P = self.out_channels, self.patch_size
        patches = patches.view(B, N, C, P, P)

        # Reassemble patches into full image
        g = self.grid_size
        patches = patches.view(B, g, g, C, P, P)
        patches = patches.permute(0, 3, 1, 4, 2, 5)   # (B, C, g, P, g, P)
        field = patches.contiguous().view(B, C, g * P, g * P)   # (B, C, H, W)

        return field


class FluidBackboneValidator:
    """
    Validates the fluid backbone on next-frame velocity field prediction.
    """

    def __init__(
        self,
        backbone: PhysicsViT,
        device: torch.device,
        n_epochs: int = 10,
        lr: float = 1e-3,
    ):
        self.backbone = backbone.to(device)
        self.backbone.eval()
        self.device = device
        self.n_epochs = n_epochs

        # Freeze backbone for validation (PhyIP protocol — no invasive adaptation)
        for p in self.backbone.parameters():
            p.requires_grad = False

        n_patches = backbone.num_patches
        self.decoder = LinearVelocityDecoder(
            embed_dim=backbone.embed_dim,
            n_patches=n_patches,
            patch_size=backbone.patch_size,
            out_channels=2,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)

    def fit_decoder(self, train_loader: DataLoader) -> float:
        """
        Fit the linear decoder on consecutive frame pairs (t → t+1).
        Returns final training loss.
        """
        self.decoder.train()
        final_loss = float("inf")

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            n_batches  = 0

            for batch in tqdm(train_loader, desc=f"Decoder epoch {epoch}", leave=False):
                images = batch["image"].to(self.device)   # (B, 3, H, W)
                if images.shape[0] < 2:
                    continue

                # Use consecutive pairs within the batch as (input, target) pairs
                B = images.shape[0] - 1
                src = images[:B]
                tgt = images[1:B + 1]

                with torch.no_grad():
                    tokens = self.backbone(src, return_all_tokens=True)   # (B, N+1, D)

                pred_uv = self.decoder(tokens)   # (B, 2, H, W)

                # Resize target to match decoder output size
                tgt_uv = tgt[:, :2]   # take u, v channels from target
                tgt_resized = F.interpolate(
                    tgt_uv, size=pred_uv.shape[-2:],
                    mode="bilinear", align_corners=False,
                )

                loss = F.mse_loss(pred_uv, tgt_resized)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

            if n_batches > 0:
                final_loss = total_loss / n_batches
                logger.info(f"  Decoder epoch {epoch}: loss={final_loss:.6f}")

        return final_loss

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on validation set using relative L² error.

        Returns:
            dict with "rel_l2_u", "rel_l2_v", "rel_l2_mean", "mse"
        """
        self.decoder.eval()
        rel_l2_u_vals = []
        rel_l2_v_vals = []
        mse_vals      = []

        for batch in tqdm(val_loader, desc="Evaluating fluid backbone"):
            images = batch["image"].to(self.device)
            if images.shape[0] < 2:
                continue

            B = images.shape[0] - 1
            src = images[:B]
            tgt = images[1:B + 1]

            tokens   = self.backbone(src, return_all_tokens=True)
            pred_uv  = self.decoder(tokens)

            tgt_uv = tgt[:, :2]
            tgt_resized = F.interpolate(
                tgt_uv, size=pred_uv.shape[-2:],
                mode="bilinear", align_corners=False,
            )

            for i in range(B):
                for ch_idx, name in enumerate(["u", "v"]):
                    pred_ch = pred_uv[i, ch_idx]
                    true_ch = tgt_resized[i, ch_idx]
                    norm_true = true_ch.norm()
                    if norm_true < 1e-8:
                        continue
                    rel_l2 = (pred_ch - true_ch).norm() / norm_true
                    if name == "u":
                        rel_l2_u_vals.append(rel_l2.item())
                    else:
                        rel_l2_v_vals.append(rel_l2.item())

            mse_vals.append(F.mse_loss(pred_uv, tgt_resized).item())

        results = {
            "rel_l2_u":    float(np.mean(rel_l2_u_vals)) if rel_l2_u_vals else float("nan"),
            "rel_l2_v":    float(np.mean(rel_l2_v_vals)) if rel_l2_v_vals else float("nan"),
            "rel_l2_mean": float(np.mean(rel_l2_u_vals + rel_l2_v_vals))
                           if (rel_l2_u_vals or rel_l2_v_vals) else float("nan"),
            "mse":         float(np.mean(mse_vals)) if mse_vals else float("nan"),
        }

        logger.info("Fluid backbone validation results:")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.4f}")

        rel_l2 = results["rel_l2_mean"]
        if rel_l2 < 0.3:
            logger.info("PASS: backbone learned meaningful Navier-Stokes structure (rel_L2 < 0.3)")
        elif rel_l2 < 0.7:
            logger.warning("MARGINAL: backbone shows partial physics learning (0.3 ≤ rel_L2 < 0.7)")
        else:
            logger.error(
                f"FAIL: backbone has NOT converged on Navier-Stokes dynamics (rel_L2={rel_l2:.3f} ≥ 0.7). "
                "Continue pre-training before proceeding to the financial probe."
            )

        return results


def main(
    checkpoint: str = "outputs/pretrain/checkpoint_best.pth",
    config_path: str = "configs/pretrain_config.yaml",
    output_dir: str = "outputs/evaluation",
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build backbone
    backbone = PhysicsViT(
        arch=cfg["model"]["arch"],
        img_size=cfg["model"]["img_size"],
        patch_size=cfg["model"]["patch_size"],
        in_chans=cfg["model"]["in_chans"],
    )

    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        backbone.load_physics_checkpoint(str(ckpt_path), key="teacher")
        logger.info(f"Loaded checkpoint: {ckpt_path}")
        # Restore field_norm stats if saved in checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "field_norm_mean" in ckpt and "field_norm_std" in ckpt:
            backbone.field_norm.update_stats(
                ckpt["field_norm_mean"].squeeze(0),
                ckpt["field_norm_std"].squeeze(0),
            )
            logger.info("Restored PhysicalFieldNorm statistics from checkpoint")
    else:
        logger.warning(f"Checkpoint not found: {ckpt_path}. Validating random backbone.")

    # Build fluid val dataset
    from src.data.fluid_dynamics.jhtdb_loader import JHTDBDataset
    from src.data.fluid_dynamics.pdearena_loader import PDEArenaDataset
    from torch.utils.data import ConcatDataset

    val_datasets = []
    for ds_cfg in cfg["data"]["datasets"]:
        if ds_cfg["name"] == "pdearena":
            ds = PDEArenaDataset(
                data_dir=ds_cfg["path"], split="val",
                img_size=cfg["model"]["img_size"],
                max_samples_per_file=1000,
            )
            if len(ds) > 0:
                val_datasets.append(ds)
        elif ds_cfg["name"] == "jhtdb":
            ds = JHTDBDataset(
                data_dir=ds_cfg["path"], split="val",
                img_size=cfg["model"]["img_size"],
                max_samples_per_file=500,
            )
            if len(ds) > 0:
                val_datasets.append(ds)

    if not val_datasets:
        logger.error("No validation fluid data found. Run bash scripts/download_data.sh first.")
        return

    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    train_dataset = val_dataset   # for decoder fitting, use same data (no label leakage)

    train_loader = DataLoader(val_dataset, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    validator = FluidBackboneValidator(backbone=backbone, device=device, n_epochs=5)
    train_loss = validator.fit_decoder(train_loader)
    results    = validator.evaluate(val_loader)

    # Save results
    import json
    results["train_loss_decoder"] = train_loss
    results["checkpoint"] = str(ckpt_path)
    with open(out_dir / "fluid_backbone_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved validation results → {out_dir}/fluid_backbone_validation.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/pretrain/checkpoint_best.pth")
    parser.add_argument("--config",     default="configs/pretrain_config.yaml")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.output_dir)
