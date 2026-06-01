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

    def _build_index(self):
        if not HDF5_AVAILABLE:
            logger.error("h5py required — install with: pip install h5py")
            return

        # Recurse into data_dir looking for HDF5 files matching the split
        patterns = [
            str(self.data_dir / "**" / f"*{self.split}*.h5"),
            str(self.data_dir / "**" / f"*{self.split}*.hdf5"),
        ]
        h5_files = []
        for pat in patterns:
            h5_files += glob.glob(pat, recursive=True)
        h5_files = sorted(set(h5_files))

        if not h5_files:
            logger.warning(
                f"No HDF5 files found for split='{self.split}' in {self.data_dir}. "
                "Run: bash scripts/download_data.sh --pdearena"
            )
            return

        for fpath in h5_files:
            try:
                with h5py.File(fpath, "r") as hf:
                    # Determine available channel arrays
                    available = {}
                    for ch in self.channels:
                        for key in [ch, f"field_{ch}", ch.upper()]:
                            if key in hf:
                                available[ch] = key
                                break

                    if not available:
                        logger.warning(f"  No matching channels in {fpath} (keys: {list(hf.keys())})")
                        continue

                    # Shape: (N_traj, T_steps, H, W)
                    ref_key = list(available.values())[0]
                    shape = hf[ref_key].shape
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
                                "available":  available,
                                "traj_idx":   i,
                                "frame_idx":  t,
                                "n_time":     n_time,
                            })
                            count += 1
                            if self.max_samples_per_file and count >= self.max_samples_per_file:
                                break
                        if self.max_samples_per_file and count >= self.max_samples_per_file:
                            break

                    logger.info(f"  PDEArena {self.split}: {fpath} → {count} frames")
            except Exception as e:
                logger.error(f"  Error indexing {fpath}: {e}")

        logger.info(f"PDEArenaDataset [{self.split}]: {len(self._index)} total frames")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        info = self._index[idx]

        channels = []
        with h5py.File(info["file"], "r") as hf:
            for ch in self.channels:
                key = info["available"].get(ch)
                if key is None:
                    channels.append(np.zeros((self.img_size, self.img_size), dtype=np.float32))
                    continue

                arr = hf[key]
                if arr.ndim == 4:
                    frame = arr[info["traj_idx"], info["frame_idx"]]   # (H, W)
                elif arr.ndim == 3:
                    frame = arr[info["frame_idx"]]
                else:
                    frame = arr[:]

                channels.append(frame.astype(np.float32))

        # Stack and resize to (3, img_size, img_size)
        img = np.stack(channels, axis=0)   # (3, H, W)
        t = torch.from_numpy(img).unsqueeze(0)
        t = F.interpolate(t, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False).squeeze(0)

        if self.normalize:
            for c in range(t.shape[0]):
                ch = t[c]
                mn, mx = ch.min(), ch.max()
                if mx > mn:
                    t[c] = (ch - mn) / (mx - mn)

        return {
            "image": t,
            "meta": {
                "file":       info["file"],
                "traj_idx":   info["traj_idx"],
                "frame_idx":  info["frame_idx"],
            },
        }
