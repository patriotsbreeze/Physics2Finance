"""
Vision Transformer backbone for the Navier-Stokes foundation model.

Uses the timm library's ViT implementation with modifications for
physical field input (continuous-valued 3-channel tensors rather than
discrete pixel intensities). Supports vit_small, vit_base, vit_large.

The encoder is designed to be frozen after pre-training for the
Non-Invasive Physical Probe (PhyIP) protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict, Any, List
from loguru import logger


class PhysicsViT(nn.Module):
    """
    Vision Transformer encoder pre-trained on Navier-Stokes fluid data.

    Wraps a timm ViT with:
    - Physical field normalization (per-channel statistics, not ImageNet)
    - CLS token extraction for dense downstream embedding
    - Patch-level embedding access for spatial analysis
    - Freeze/unfreeze support for PhyIP protocol enforcement
    """

    ARCH_CONFIGS = {
        "vit_tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
        "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    }

    def __init__(
        self,
        arch: str = "vit_base_patch16_224",
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        super().__init__()
        self.arch = arch
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.encoder = timm.create_model(
            arch,
            pretrained=pretrained,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # headless — features only
            global_pool="",  # return all tokens
        )

        self.embed_dim = self.encoder.embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Physical field normalization (trained from fluid data statistics)
        self.field_norm = PhysicalFieldNorm(in_chans)

        if frozen:
            self.freeze()

        logger.info(
            f"PhysicsViT [{arch}]: embed_dim={self.embed_dim}, "
            f"patches={self.num_patches}, frozen={frozen}"
        )

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) physical field tensor.
            return_all_tokens: if True, return all patch tokens (B, N+1, D);
                               if False, return only CLS token (B, D).

        Returns:
            Dense embedding tensor.
        """
        x = self.field_norm(x)
        tokens = self.encoder.forward_features(x)  # (B, N+1, D)

        if return_all_tokens:
            return tokens

        # CLS token
        return tokens[:, 0]

    def get_intermediate_layers(
        self, x: torch.Tensor, n: int = 4
    ) -> List[torch.Tensor]:
        """Return feature maps from the last n transformer blocks."""
        x = self.field_norm(x)
        x = self.encoder.patch_embed(x)

        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)

        outputs = []
        blocks = list(self.encoder.blocks)
        for i, block in enumerate(blocks):
            x = block(x)
            if i >= len(blocks) - n:
                outputs.append(x)

        return outputs

    def freeze(self):
        """Freeze all encoder weights (PhyIP protocol)."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("PhysicsViT: encoder FROZEN (PhyIP mode)")

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        logger.info("PhysicsViT: encoder UNFROZEN")

    def load_physics_checkpoint(self, checkpoint_path: str, key: str = "teacher"):
        """Load weights from a DINO-style checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if key in ckpt:
            state_dict = ckpt[key]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Strip DINO head prefix if present
        state_dict = {
            k.replace("backbone.", "encoder."): v
            for k, v in state_dict.items()
            if not k.startswith("head.")
        }

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(
            f"Loaded checkpoint: {checkpoint_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )


class PhysicalFieldNorm(nn.Module):
    """
    Learnable per-channel normalization for physical field inputs.

    Initialized with mean=0, std=1 (identity transform). During
    pre-training on fluid data, the running statistics adapt to the
    actual distribution of u, v, p values.
    """

    def __init__(self, num_channels: int = 3):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("std", torch.ones(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)

    def update_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean.copy_(mean.view(1, -1, 1, 1))
        self.std.copy_(std.view(1, -1, 1, 1))


class DINOHead(nn.Module):
    """
    DINO projection head mapping ViT embeddings to a high-dimensional
    prototype space for the self-supervised contrastive loss.

    Architecture: Linear → BN → GELU → Linear → L2 norm → Linear
    """

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        use_bn: bool = False,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
