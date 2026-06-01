"""
Phase I: Self-supervised pre-training of the foundation model on fluid dynamics data.

Uses DINOv3-style self-distillation with local/global crop augmentation.
The student processes both local and global views; the teacher (EMA copy)
processes only global views.

WARNING: This is computationally expensive. On an A100 GPU:
  - vit_small: ~24h for 100 epochs
  - vit_base:  ~72h for 100 epochs
Run with: python -m src.training.pretrain_fluid --config configs/pretrain_config.yaml
"""

import os
import copy
import math
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from loguru import logger
import yaml

from src.models.vit_backbone import PhysicsViT, DINOHead
from src.models.dino_loss import DINOLoss, MomentumUpdater
from src.data.fluid_dynamics.jhtdb_loader import JHTDBDataset
from src.data.fluid_dynamics.pdearena_loader import PDEArenaDataset
from src.data.fluid_dynamics.preprocessing import MultiScalePatchExtractor


def build_dataset(cfg: dict, split: str = "train") -> ConcatDataset:
    datasets = []
    data_cfg = cfg["data"]

    for ds_cfg in data_cfg["datasets"]:
        name = ds_cfg["name"]
        path = ds_cfg["path"]

        if name == "pdearena":
            ds = PDEArenaDataset(
                data_dir=path,
                split=split,
                img_size=data_cfg["img_size"],
            )
            datasets.append(ds)
        elif name == "jhtdb":
            ds = JHTDBDataset(
                data_dir=path,
                split=split,
                img_size=data_cfg["img_size"],
            )
            datasets.append(ds)

    if not datasets:
        raise RuntimeError("No datasets found. Run scripts/download_data.sh first.")

    return ConcatDataset(datasets)


class DINOCollate:
    """Applies multi-scale crop augmentation, returning list of crop tensors."""

    def __init__(self, extractor: MultiScalePatchExtractor):
        self.extractor = extractor

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        all_crops = []
        for img in images:
            crops = self.extractor(img)
            all_crops.append(crops)

        # Transpose: list of images × crops → list of crops × batch
        n_crops = len(all_crops[0])
        batched_crops = [
            torch.stack([all_crops[i][j] for i in range(len(all_crops))])
            for j in range(n_crops)
        ]
        return batched_crops


def get_cosine_schedule_with_warmup(
    optimizer, warmup_epochs: int, total_epochs: int, min_lr: float, base_lr: float
):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr / base_lr + 0.5 * (1 - min_lr / base_lr) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    student_head: nn.Module,
    teacher_head: nn.Module,
    criterion: DINOLoss,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    epoch: int,
    cfg: dict,
    momentum_updater: MomentumUpdater,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    student.train()
    student_head.train()
    teacher.eval()
    teacher_head.eval()

    total_loss = 0.0
    n_batches = len(loader)
    total_steps = n_batches * cfg["training"]["epochs"]
    step = epoch * n_batches

    for batch_idx, crops in enumerate(loader):
        # crops: list of (B, 3, H, W) tensors — [global1, global2, local1..localk]
        n_global = 2
        global_crops = torch.cat(crops[:n_global], dim=0).to(device)
        all_crops = torch.cat(crops, dim=0).to(device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # Teacher forward on global crops only
            with torch.no_grad():
                teacher_feats = teacher(global_crops)
                teacher_out = [teacher_head(teacher_feats[:len(crops[0])]),
                               teacher_head(teacher_feats[len(crops[0]):])]

            # Student forward on all crops
            student_feats = student(all_crops)
            B = len(crops[0])
            student_out = [student_head(student_feats[i * B:(i + 1) * B]) for i in range(len(crops))]

            loss = criterion(student_out, teacher_out, epoch)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(student_head.parameters()),
                cfg["training"]["clip_grad"],
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(student_head.parameters()),
                cfg["training"]["clip_grad"],
            )
            optimizer.step()

        # Freeze last layer during warmup
        if epoch < cfg["dino"]["freeze_last_layer"] if "freeze_last_layer" in cfg["dino"] else 1:
            for n, p in student_head.named_parameters():
                if "last_layer" in n:
                    p.grad = None

        # Update teacher via EMA
        m = momentum_updater.get_momentum(step + batch_idx, total_steps)
        momentum_updater.update(student, teacher, m)
        momentum_updater.update(student_head, teacher_head, m)

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{n_batches}] "
                f"loss={loss.item():.4f} momentum={m:.4f}"
            )

    return total_loss / n_batches


def main(config_path: str = "configs/pretrain_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Build datasets
    train_dataset = build_dataset(cfg, split="train")
    val_dataset = build_dataset(cfg, split="val")

    extractor = MultiScalePatchExtractor(
        img_size=cfg["model"]["img_size"],
        global_crops_scale=cfg["dino"]["global_crops_scale"],
        local_crops_scale=cfg["dino"]["local_crops_scale"],
        n_local_crops=cfg["dino"]["local_crops_number"],
    )
    collate_fn = DINOCollate(extractor)

    n_crops = 2 + cfg["dino"]["local_crops_number"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Build student and teacher (same architecture, EMA weights)
    student = PhysicsViT(
        arch=cfg["model"]["arch"],
        img_size=cfg["model"]["img_size"],
        patch_size=cfg["model"]["patch_size"],
        in_chans=cfg["model"]["in_chans"],
    ).to(device)

    teacher = copy.deepcopy(student)
    teacher.freeze()

    student_head = DINOHead(
        in_dim=student.embed_dim,
        out_dim=cfg["dino"]["out_dim"],
    ).to(device)

    teacher_head = DINOHead(
        in_dim=teacher.embed_dim,
        out_dim=cfg["dino"]["out_dim"],
    ).to(device)
    for p in teacher_head.parameters():
        p.requires_grad = False

    criterion = DINOLoss(
        out_dim=cfg["dino"]["out_dim"],
        n_crops=n_crops,
        teacher_temp=cfg["dino"]["teacher_temp"],
        student_temp=cfg["dino"]["student_temp"],
        center_momentum=cfg["dino"]["center_momentum"],
        n_epochs=cfg["training"]["epochs"],
    ).to(device)

    params = list(student.parameters()) + list(student_head.parameters())
    optimizer = optim.AdamW(
        params,
        lr=cfg["training"]["base_lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=cfg["training"]["warmup_epochs"],
        total_epochs=cfg["training"]["epochs"],
        min_lr=cfg["training"]["min_lr"],
        base_lr=cfg["training"]["base_lr"],
    )
    momentum_updater = MomentumUpdater(
        base_momentum=cfg["dino"]["momentum_teacher"],
    )
    scaler = torch.cuda.amp.GradScaler() if (cfg["hardware"]["fp16"] and device.type == "cuda") else None

    # Training loop
    best_loss = float("inf")
    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_one_epoch(
            student, teacher, student_head, teacher_head,
            criterion, optimizer, train_loader, epoch, cfg,
            momentum_updater, device, scaler,
        )
        scheduler.step()

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        if epoch % cfg["training"]["save_every"] == 0 or train_loss < best_loss:
            ckpt = {
                "epoch": epoch,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "student_head": student_head.state_dict(),
                "teacher_head": teacher_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": train_loss,
                "cfg": cfg,
            }
            torch.save(ckpt, output_dir / f"checkpoint_epoch{epoch:04d}.pth")
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(ckpt, output_dir / "checkpoint_best.pth")
                logger.info(f"New best checkpoint: loss={best_loss:.4f}")

    logger.info("Pre-training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_config.yaml")
    args = parser.parse_args()
    main(args.config)
