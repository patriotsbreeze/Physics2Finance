"""
PDEArena dataset loader for NavierStokes-2D forced turbulence.

PDEArena hosts NavierStokes-2D (forced turbulence, incompressible) as HDF5 files
on HuggingFace: https://huggingface.co/datasets/pdearena/NavierStokes-2D

Each HDF5 file contains:
  u:  (N, T, H, W) x-velocity
  v:  (N, T, H, W) y-velocity
  p:  (N, T, H, W) pressure       (if present)
  vx: alias for u in some files

We emit random (u, v, p) frames resized to (3, img_size, img_size).
The three channels map directly to the R/G/B → u/v/p analogy used
throughout the project plan.

Download with:
  huggingface-cli download pdearena/NavierStokes-2D --repo-type dataset \
    --local-dir data/pdearena/NavierStokes2D-ForcedTurbulence
"""

import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logger.warning("h5py not installed: pip install h5py")


class PDEArenaDataset(Dataset):
    """
    Dataset over PDEArena NavierStokes-2D HDF5 snapshots.

    Each sample is a dict:
      "image": (3, img_size, img_size) float32 tensor — [u, v, p] channels
      "meta":  dict with dataset file and frame indices
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 224,
        channels: List[str] = ("u", "v", "p"),
        max_samples_per_file: Optional[int] = None,
        normalize: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.channels = list(channels)
        self.max_samples_per_file = max_samples_per_file
        self.normalize = normalize

        self._index: List[dict] = []   # list of {file, traj_idx, frame_idx}
        self._build_index()

    # PDEArena NavierStokes-2D actual field names (vx=x-velocity, vy=y-velocity, u=buoyancy)
    # Maps our logical channel names to actual HDF5 dataset names inside the split group
    _CHANNEL_MAP = {
        "u":  ["vx", "u_x", "u"],    # x-velocity (try vx first)
        "v":  ["vy", "u_y", "v"],    # y-velocity
        "p":  ["u",  "p",   "buo_y"],# buoyancy/pressure scalar
    }

    def _build_index(self):
        if not HDF5_AVAILABLE:
            logger.error("h5py required — install with: pip install h5py")
            return

        # PDEArena files are named NavierStokes2D_{split}_*.h5
        # The split name in filenames uses "valid" not "val"
        split_tag = "valid" if self.split == "val" else self.split
        patterns = [
            str(self.data_dir / "**" / f"*{split_tag}*.h5"),
            str(self.data_dir / "**" / f"*{split_tag}*.hdf5"),
        ]
        h5_files = []
        for pat in patterns:
            h5_files += glob.glob(pat, recursive=True)
        h5_files = sorted(set(h5_files))

        if not h5_files:
            logger.warning(
                f"No HDF5 files found for split='{self.split}' (tag='{split_tag}') "
                f"in {self.data_dir}. Run: bash scripts/download_data.sh --pdearena"
            )
            return

        for fpath in h5_files:
            try:
                with h5py.File(fpath, "r") as hf:
                    # PDEArena files have a top-level group named after the split
                    # e.g. hf["train"]["vx"] shape (N, T, H, W)
                    group = None
                    for gname in [split_tag, self.split, "data"]:
                        if gname in hf:
                            group = hf[gname]
                            break
                    if group is None:
                        # Flat file — try top-level directly
                        group = hf

                    group_keys = list(group.keys())

                    # Resolve logical channel → actual key
                    available = {}
                    for logical_ch in self.channels:
                        candidates = self._CHANNEL_MAP.get(logical_ch, [logical_ch])
                        for cand in candidates:
                            if cand in group_keys and hasattr(group[cand], "shape"):
                                available[logical_ch] = cand
                                break

                    if not available:
                        logger.warning(
                            f"  No matching channels in {fpath} "
                            f"(group keys: {group_keys}, wanted: {self.channels})"
                        )
                        continue

                    ref_key = list(available.values())[0]
                    shape = group[ref_key].shape   # (N_traj, T_steps, H, W)
                    if len(shape) == 4:
                        n_traj, n_time = shape[0], shape[1]
                    elif len(shape) == 3:
                        n_traj, n_time = 1, shape[0]
                    else:
                        continue

                    count = 0
                    for i in range(n_traj):
                        for t in range(n_time):
                            self._index.append({
                                "file":       fpath,
                                "group_name": split_tag if split_tag in hf else None,
                                "available":  available,
                                "traj_idx":   i,
                                "frame_idx":  t,
                            })
                            count += 1
                            if self.max_samples_per_file and count >= self.max_samples_per_file:
                                break
                        if self.max_samples_per_file and count >= self.max_samples_per_file:
                            break

                    logger.info(f"  PDEArena {self.split}: {fpath} → {count} frames "
                                f"({n_traj} traj × {n_time} steps)")
            except Exception as e:
                logger.error(f"  Error indexing {fpath}: {e}")

        logger.info(f"PDEArenaDataset [{self.split}]: {len(self._index)} total frames")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        info = self._index[idx]

        channels = []
        with h5py.File(info["file"], "r") as hf:
            # Navigate to the split group if present
            group_name = info.get("group_name")
            node = hf[group_name] if (group_name and group_name in hf) else hf

            for ch in self.channels:
                key = info["available"].get(ch)
                if key is None or key not in node:
                    channels.append(np.zeros((self.img_size, self.img_size), dtype=np.float32))
                    continue

                arr = node[key]
                if arr.ndim == 4:
                    frame = arr[info["traj_idx"], info["frame_idx"]]   # (H, W)
                elif arr.ndim == 3:
                    frame = arr[info["frame_idx"]]
                else:
                    frame = arr[:]

                channels.append(np.array(frame, dtype=np.float32))

        # Stack and resize to (3, img_size, img_size)
        img = np.stack(channels, axis=0)   # (3, H, W)
        t = torch.from_numpy(img).unsqueeze(0)
        t = F.interpolate(t, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False).squeeze(0)

        if self.normalize:
            for c in range(t.shape[0]):
                ch = t[c]
                mu = ch.mean()
                sigma = ch.std()
                if sigma > 1e-6:
                    lo = mu - 3.0 * sigma
                    hi = mu + 3.0 * sigma
                    t[c] = 2.0 * (ch.clamp(lo, hi) - lo) / (hi - lo) - 1.0
                else:
                    t[c] = torch.zeros_like(ch)

        return {
            "image": t,
            "meta": {
                "file":       info["file"],
                "traj_idx":   info["traj_idx"],
                "frame_idx":  info["frame_idx"],
            },
        }
