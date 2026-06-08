"""
Phase I: Self-supervised pre-training of the foundation model on fluid dynamics data.

Uses DINOv3-style self-distillation with local/global crop augmentation.
The student processes both local and global views; the teacher (EMA copy)
processes only global views.

WARNING: This is computationally expensive. On an A100 GPU:
  - vit_small: ~24h for 100 epochs
  - vit_base:  ~72h for 100 epochs

Single-GPU:
  python -m src.training.pretrain_fluid --config configs/pretrain_config.yaml

Multi-GPU (DDP via torchrun):
  torchrun --nproc_per_node=4 -m src.training.pretrain_fluid --config configs/pretrain_config.yaml
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from loguru import logger
import yaml

from src.models.vit_backbone import PhysicsViT, DINOHead
from src.models.dino_loss import DINOLoss, MomentumUpdater
from src.data.fluid_dynamics.jhtdb_loader import JHTDBDataset
from src.data.fluid_dynamics.pdearena_loader import PDEArenaDataset
from src.data.fluid_dynamics.preprocessing import MultiScalePatchExtractor, compute_dataset_channel_stats

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    """Initialize distributed training if launched via torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        logger.info(f"DDP: rank={rank}/{world_size}, local_rank={local_rank}")
        return local_rank, world_size
    return 0, 1


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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
    # ── DDP setup ─────────────────────────────────────────────────────────────
    local_rank, world_size = setup_ddp()
    is_main = is_main_process()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["training"]["output_dir"])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    if dist.is_available() and dist.is_initialized():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device} (world_size={world_size})")

    # ── W&B initialization ────────────────────────────────────────────────────
    use_wandb = WANDB_AVAILABLE and cfg.get("logging", {}).get("wandb", False) and is_main
    if use_wandb:
        wandb.init(
            project=cfg.get("logging", {}).get("wandb_project", "Physics2Finance"),
            name=cfg.get("logging", {}).get("run_name", "pretrain_fluid"),
            config=cfg,
        )
        logger.info("W&B initialized")

    # ── Build datasets ────────────────────────────────────────────────────────
    train_dataset = build_dataset(cfg, split="train")
    val_dataset   = build_dataset(cfg, split="val")

    # ── Compute PhysicalFieldNorm statistics (gap #13 fix) ────────────────────
    if is_main:
        logger.info("Computing dataset channel statistics for PhysicalFieldNorm...")
        norm_mean, norm_std = compute_dataset_channel_stats(
            train_dataset, n_samples=500
        )
        logger.info(f"  Channel mean: {norm_mean.squeeze()}")
        logger.info(f"  Channel std:  {norm_std.squeeze()}")
    else:
        norm_mean = torch.zeros(1, 3, 1, 1)
        norm_std  = torch.ones(1, 3, 1, 1)

    if dist.is_available() and dist.is_initialized():
        dist.broadcast(norm_mean, src=0)
        dist.broadcast(norm_std,  src=0)

    extractor  = MultiScalePatchExtractor(
        img_size=cfg["model"]["img_size"],
        global_crops_scale=cfg["dino"]["global_crops_scale"],
        local_crops_scale=cfg["dino"]["local_crops_scale"],
        n_local_crops=cfg["dino"]["local_crops_number"],
    )
    collate_fn = DINOCollate(extractor)
    n_crops    = 2 + cfg["dino"]["local_crops_number"]

    # ── DataLoaders with DistributedSampler ───────────────────────────────────
    train_sampler = DistributedSampler(train_dataset, shuffle=True) \
        if (dist.is_available() and dist.is_initialized()) else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # ── Build student and teacher ─────────────────────────────────────────────
    student = PhysicsViT(
        arch=cfg["model"]["arch"],
        img_size=cfg["model"]["img_size"],
        patch_size=cfg["model"]["patch_size"],
        in_chans=cfg["model"]["in_chans"],
    ).to(device)

    # Apply computed channel statistics to PhysicalFieldNorm (gap #13 fix)
    student.field_norm.update_stats(norm_mean.squeeze(0), norm_std.squeeze(0))

    teacher = copy.deepcopy(student)
    teacher.freeze()

    # Wrap student in DDP
    if dist.is_available() and dist.is_initialized():
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=False)
        student_module = student.module
    else:
        student_module = student

    student_head = DINOHead(
        in_dim=student_module.embed_dim,
        out_dim=cfg["dino"]["out_dim"],
    ).to(device)
    if dist.is_available() and dist.is_initialized():
        student_head = DDP(student_head, device_ids=[local_rank])

    teacher_head = DINOHead(
        in_dim=teacher.embed_dim,
        out_dim=cfg["dino"]["out_dim"],
    ).to(device)
    for p in teacher_head.parameters():
        p.requires_grad = False

    criterion = DINOLoss(
        out_dim=cfg["dino"]["out_dim"],
        n_crops=n_crops,
        warmup_teacher_temp=cfg["dino"]["warmup_teacher_temp"],
        teacher_temp=cfg["dino"]["teacher_temp"],
        warmup_teacher_temp_epochs=cfg["dino"]["warmup_teacher_temp_epochs"],
        student_temp=cfg["dino"]["student_temp"],
        center_momentum=cfg["dino"]["center_momentum"],
        n_epochs=cfg["training"]["epochs"],
    ).to(device)

    params = (
        list(student.parameters()) + list(student_head.parameters())
    )
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
    momentum_updater = MomentumUpdater(base_momentum=cfg["dino"]["momentum_teacher"])
    scaler = torch.cuda.amp.GradScaler() \
        if (cfg["hardware"]["fp16"] and device.type == "cuda") else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(cfg["training"]["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            student, teacher, student_head, teacher_head,
            criterion, optimizer, train_loader, epoch, cfg,
            momentum_updater, device, scaler,
        )
        scheduler.step()

        if is_main:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            if use_wandb:
                wandb.log({"train/loss": train_loss, "epoch": epoch,
                           "lr": optimizer.param_groups[0]["lr"]})

            if epoch % cfg["training"]["save_every"] == 0 or train_loss < best_loss:
                # Save teacher backbone (no DDP wrapper)
                teacher_state = teacher.state_dict() \
                    if not isinstance(teacher, DDP) else teacher.module.state_dict()
                student_state = student.state_dict() \
                    if not isinstance(student, DDP) else student.module.state_dict()
                ckpt = {
                    "epoch":        epoch,
                    "student":      student_state,
                    "teacher":      teacher_state,
                    "student_head": student_head.state_dict()
                        if not isinstance(student_head, DDP) else student_head.module.state_dict(),
                    "teacher_head": teacher_head.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "loss":         train_loss,
                    "cfg":          cfg,
                    "field_norm_mean": norm_mean.cpu(),
                    "field_norm_std":  norm_std.cpu(),
                }
                torch.save(ckpt, output_dir / f"checkpoint_epoch{epoch:04d}.pth")
                if train_loss < best_loss:
                    best_loss = train_loss
                    torch.save(ckpt, output_dir / "checkpoint_best.pth")
                    logger.info(f"New best checkpoint: loss={best_loss:.4f}")
                    if use_wandb:
                        wandb.run.summary["best_loss"] = best_loss

    if is_main:
        logger.info("Pre-training complete.")
        if use_wandb:
            wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_config.yaml")
    args = parser.parse_args()
    main(args.config)
