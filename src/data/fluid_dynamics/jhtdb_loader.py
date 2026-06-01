"""
Johns Hopkins Turbulent Database (JHTDB) dataset loader.

JHTDB provides DNS simulation of forced isotropic turbulence at Re_λ≈433
on a 1024³ periodic grid.  The public dataset is available as HDF5 cutouts
downloadable from:
  http://turbulence.pha.jhu.edu/cutouts/

The demo cutout (sample_iso1024.h5) is downloaded by scripts/download_data.sh.
Full access requires registration at turbulence.pha.jhu.edu.

HDF5 structure (JHTDB isotropic cutout):
  /u:  (T, Nx, Ny, Nz, 3) velocity vector field   — or
  /velocity: same layout
  /p:  (T, Nx, Ny, Nz)    pressure scalar

We take 2D cross-sectional slices (z=0 plane by default) to produce
(u_x, u_y, p) frames compatible with the 3-channel ViT input.
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


class JHTDBDataset(Dataset):
    """
    Dataset over JHTDB velocity/pressure HDF5 cutout files.

    Each sample is a dict:
      "image": (3, img_size, img_size) float32 tensor — [u_x, u_y, p]
      "meta":  dict with file path and frame/slice indices
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 224,
        slice_axis: int = 2,           # which spatial axis to take 2D cross-section on
        max_samples_per_file: Optional[int] = None,
        normalize: bool = True,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.slice_axis = slice_axis
        self.max_samples_per_file = max_samples_per_file
        self.normalize = normalize
        self.train_frac = train_frac
        self.val_frac = val_frac

        self._index: List[dict] = []
        self._build_index()

    def _build_index(self):
        if not HDF5_AVAILABLE:
            logger.error("h5py required — install with: pip install h5py")
            return

        h5_files = sorted(
            glob.glob(str(self.data_dir / "**" / "*.h5"),  recursive=True) +
            glob.glob(str(self.data_dir / "**" / "*.hdf5"), recursive=True)
        )

        if not h5_files:
            logger.warning(
                f"No HDF5 files found in {self.data_dir}. "
                "Run: bash scripts/download_data.sh --jhtdb"
            )
            return

        for fpath in h5_files:
            try:
                self._index_file(fpath)
            except Exception as e:
                logger.error(f"Error indexing JHTDB file {fpath}: {e}")

        # Split index by fraction
        n = len(self._index)
        if self.split == "train":
            self._index = self._index[:int(self.train_frac * n)]
        elif self.split == "val":
            s = int(self.train_frac * n)
            e = int((self.train_frac + self.val_frac) * n)
            self._index = self._index[s:e]
        else:
            self._index = self._index[int((self.train_frac + self.val_frac) * n):]

        logger.info(f"JHTDBDataset [{self.split}]: {len(self._index)} frames")

    def _index_file(self, fpath: str):
        with h5py.File(fpath, "r") as hf:
            # Find velocity and pressure keys
            vel_key = None
            for k in ["u", "velocity", "vel", "U"]:
                if k in hf:
                    vel_key = k
                    break

            pres_key = None
            for k in ["p", "pressure", "P"]:
                if k in hf:
                    pres_key = k
                    break

            if vel_key is None:
                logger.warning(f"  No velocity field found in {fpath} (keys: {list(hf.keys())})")
                return

            vel_shape = hf[vel_key].shape   # e.g. (T, Nx, Ny, Nz, 3) or (T, H, W, 3)

            if len(vel_shape) == 5:
                n_time, nx, ny, nz, _ = vel_shape
                # For each time step, offer all z-slices
                n_slices = nz if self.slice_axis == 2 else (ny if self.slice_axis == 1 else nx)
            elif len(vel_shape) == 4:
                n_time, nx, ny, _ = vel_shape
                n_slices = 1
            elif len(vel_shape) == 3:
                # (T, H, W) — single channel, treat as single frame
                n_time = vel_shape[0]
                n_slices = 1
            else:
                return

            count = 0
            for t in range(n_time):
                for s in range(min(n_slices, 8)):   # limit slices per time step
                    self._index.append({
                        "file":      fpath,
                        "vel_key":   vel_key,
                        "pres_key":  pres_key,
                        "time_idx":  t,
                        "slice_idx": s,
                        "vel_ndim":  len(vel_shape),
                    })
                    count += 1
                    if self.max_samples_per_file and count >= self.max_samples_per_file:
                        return

            logger.info(f"  JHTDB: {fpath} → {count} frames")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        info = self._index[idx]

        with h5py.File(info["file"], "r") as hf:
            vel_arr  = hf[info["vel_key"]]
            pres_arr = hf[info["pres_key"]] if info["pres_key"] else None
            t = info["time_idx"]
            s = info["slice_idx"]

            if info["vel_ndim"] == 5:
                # (T, Nx, Ny, Nz, 3) → take z-slice s
                vel_slice = vel_arr[t, :, :, s, :]   # (Nx, Ny, 3)
                ux = vel_slice[:, :, 0].astype(np.float32)
                uy = vel_slice[:, :, 1].astype(np.float32)
                if pres_arr is not None:
                    p = pres_arr[t, :, :, s].astype(np.float32)
                else:
                    p = np.zeros_like(ux)
            elif info["vel_ndim"] == 4:
                vel_slice = vel_arr[t, :, :, :]   # (H, W, 3)
                ux = vel_slice[:, :, 0].astype(np.float32)
                uy = vel_slice[:, :, 1].astype(np.float32)
                p = (vel_slice[:, :, 2].astype(np.float32)
                     if vel_slice.shape[2] >= 3 else np.zeros_like(ux))
            else:
                ux = vel_arr[t].astype(np.float32)
                uy = np.zeros_like(ux)
                p  = np.zeros_like(ux)

        img = np.stack([ux, uy, p], axis=0)   # (3, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.img_size, self.img_size),
                               mode="bilinear", align_corners=False).squeeze(0)

        if self.normalize:
            for c in range(tensor.shape[0]):
                ch = tensor[c]
                mn, mx = ch.min(), ch.max()
                if mx > mn:
                    tensor[c] = (ch - mn) / (mx - mn)

        return {
            "image": tensor,
            "meta": {
                "file":      info["file"],
                "time_idx":  info["time_idx"],
                "slice_idx": info["slice_idx"],
            },
        }
