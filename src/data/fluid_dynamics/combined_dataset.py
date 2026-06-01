"""
Combined fluid dynamics dataset that mixes JHTDB and PDEArena samples.

Uses weighted sampling so that each dataset contributes proportionally
to the pre-training corpus regardless of raw size differences.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import List, Tuple
from loguru import logger

from src.data.fluid_dynamics.jhtdb_loader import JHTDBDataset
from src.data.fluid_dynamics.pdearena_loader import PDEArenaDataset


class CombinedFluidDataset(Dataset):
    """
    Weighted mixture of fluid dynamics datasets for pre-training.

    Weights are specified per dataset and applied via WeightedRandomSampler,
    ensuring balanced exposure to different flow regimes and Reynolds numbers.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float],
        transform=None,
    ):
        assert len(datasets) == len(weights), "Must provide one weight per dataset"
        self.datasets = datasets
        self.weights = weights
        self.transform = transform

        self.cumulative_sizes: List[int] = []
        cumsum = 0
        for ds in datasets:
            cumsum += len(ds)
            self.cumulative_sizes.append(cumsum)

        self.total = self.cumulative_sizes[-1]
        self._sample_weights = self._build_sample_weights()

        logger.info(
            f"CombinedFluidDataset: {self.total} total samples from "
            f"{len(datasets)} datasets with weights={weights}"
        )

    def _build_sample_weights(self) -> torch.Tensor:
        weights = []
        for ds, w in zip(self.datasets, self.weights):
            w_per_sample = w / max(len(ds), 1)
            weights.extend([w_per_sample] * len(ds))
        return torch.tensor(weights, dtype=torch.float)

    def get_sampler(self) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            weights=self._sample_weights,
            num_samples=self.total,
            replacement=True,
        )

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> dict:
        # Locate which sub-dataset this index belongs to
        ds_idx = 0
        offset = 0
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                ds_idx = i
                offset = cum_size - len(self.datasets[i])
                break

        sample = self.datasets[ds_idx][idx - offset]
        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])
        return sample
