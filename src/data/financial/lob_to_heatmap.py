"""
LOB-to-heatmap encoder: physics-correct mapping of LOB snapshots to (3, H, W) images.

Channel mapping (mechanistic fluid analogy, not metaphorical):
  u (ch 0): price advection field  — dP_rel/dt, rate each price level moves vs mid
  v (ch 1): volume flux field      — dV_norm/dt, order arrival/cancellation rate per level
  p (ch 2): volume density field   — V_level/V_total, liquidity "pressure" at each level

Rationale: these three fields make LOB images interpretable by turbulence-trained ViT filters.
  - Vorticity w = du/dy - dv/dx detects order book turbulence (level shift vs volume flow)
  - Divergence div = du/dx + dv/dy captures market stress / book expansion
  - Pressure gradient dp activates the same filters as Navier-Stokes compression zones
  - A random ViT has no such structured filter bank; physics ViT does.

Heatmap layout:
  X-axis: time steps (oldest left, newest right)
  Y-axis: price levels centered on mid-price spread
    Rows 0..k-1:   ask levels (row 0 = worst/highest ask, row k-1 = best ask)
    Rows k..2k-1:  bid levels (row k = best bid, row 2k-1 = worst/lowest bid)

Input format (FI-2010 / DeepLOB convention):
  Interleaved columns: [P_ask_1, V_ask_1, P_bid_1, V_bid_1, P_ask_2, V_ask_2, ...]
  Shape: (window_size, 4 * lob_levels)
"""

import numpy as np
import torch
import torch.nn.functional as F


class LOBHeatmapEncoder:
    """
    Converts a sequence of LOB snapshots to a (3, img_size, img_size) float32 tensor.

    The three channels are fluid-dynamic quantities:
      ch 0 = u: price advection (how fast each price level moves relative to mid)
      ch 1 = v: volume flux     (order arrival/cancellation rate per level)
      ch 2 = p: pressure        (normalized volume density per level)

    All channels σ-clipped and rescaled to [-1, 1] to match physics simulation statistics.
    """

    def __init__(
        self,
        lob_levels: int = 10,
        img_size: int = 224,
        sigma_clip: float = 3.0,
    ):
        self.lob_levels = lob_levels
        self.img_size = img_size
        self.sigma_clip = sigma_clip

    def encode_matrix(self, lob_matrix: np.ndarray) -> torch.Tensor:
        """
        Args:
            lob_matrix: (T, 4*k) — interleaved [P_ask_i, V_ask_i, P_bid_i, V_bid_i]
        Returns:
            (3, img_size, img_size) float32 tensor
        """
        k = self.lob_levels
        ask_prices = lob_matrix[:, 0::4][:, :k]
        ask_vols   = lob_matrix[:, 1::4][:, :k]
        bid_prices = lob_matrix[:, 2::4][:, :k]
        bid_vols   = lob_matrix[:, 3::4][:, :k]
        return self.encode(ask_prices, ask_vols, bid_prices, bid_vols)

    def encode(
        self,
        ask_prices: np.ndarray,
        ask_vols: np.ndarray,
        bid_prices: np.ndarray,
        bid_vols: np.ndarray,
    ) -> torch.Tensor:
        """
        Args:
            ask_prices, ask_vols, bid_prices, bid_vols: each (T, k)
        Returns:
            (3, img_size, img_size) float32 tensor with channels [u, v, p]
        """
        T, k = ask_vols.shape
        eps = 1e-10
        H = 2 * k

        # Replace NaNs/infs with neutral values before any computation
        ask_prices = np.nan_to_num(ask_prices, nan=0.0, posinf=0.0, neginf=0.0)
        ask_vols   = np.nan_to_num(ask_vols,   nan=0.0, posinf=0.0, neginf=0.0)
        bid_prices = np.nan_to_num(bid_prices, nan=0.0, posinf=0.0, neginf=0.0)
        bid_vols   = np.nan_to_num(bid_vols,   nan=0.0, posinf=0.0, neginf=0.0)
        ask_vols   = np.abs(ask_vols)
        bid_vols   = np.abs(bid_vols)

        # --- Mid-price reference (T,) ---
        mid = (ask_prices[:, 0] + bid_prices[:, 0]) / 2.0
        mid = np.where(mid == 0, eps, mid)

        # --- Spatial grids (H, T) ---
        # Row 0..k-1:   ask levels, flipped so worst ask (farthest from mid) is at row 0
        # Row k..2k-1:  bid levels, best bid at row k, worst bid at row 2k-1

        # Price grid: fractional displacement from mid-price
        ask_rel = (ask_prices - mid[:, None]) / mid[:, None]   # (T, k), ≥0
        bid_rel = (bid_prices - mid[:, None]) / mid[:, None]   # (T, k), ≤0

        price_grid = np.empty((H, T), dtype=np.float64)
        price_grid[:k, :] = ask_rel[:, ::-1].T   # flip: row k-1 = best ask (near spread)
        price_grid[k:, :] = bid_rel.T            # row k   = best bid (near spread)

        # Volume grid: normalized by total depth so scale-invariant
        vol_raw = np.empty((H, T), dtype=np.float64)
        vol_raw[:k, :] = ask_vols[:, ::-1].T
        vol_raw[k:, :] = bid_vols.T

        total_depth = vol_raw.sum(axis=0, keepdims=True).clip(eps)
        vol_norm = vol_raw / total_depth   # (H, T) ∈ [0, 1]

        # ===================================================================
        # Channel u: price advection field  dP_rel/dt
        # Captures how fast each price level shifts relative to mid over time.
        # Analogue: fluid x-velocity u(y, t) — advection of price "fluid parcels".
        # Positive u: level moving up (ask levels lifting, bid levels widening).
        # ===================================================================
        u = np.zeros((H, T), dtype=np.float64)
        if T > 1:
            u[:, 1:]  = price_grid[:, 1:] - price_grid[:, :-1]
            u[:, 0]   = u[:, 1]   # copy boundary

        # ===================================================================
        # Channel v: volume flux field  dV_norm/dt
        # Captures order arrival (v > 0) and cancellation (v < 0) at each level.
        # Analogue: fluid y-velocity v(y, t) — cross-stream volume flow.
        # Together with u, their curl gives vorticity = order book turbulence.
        # ===================================================================
        v = np.zeros((H, T), dtype=np.float64)
        if T > 1:
            v[:, 1:]  = vol_norm[:, 1:] - vol_norm[:, :-1]
            v[:, 0]   = v[:, 1]

        # ===================================================================
        # Channel p: volume density field  V_level / V_total
        # Instantaneous normalized volume — liquidity "pressure" at each level.
        # Analogue: fluid pressure p(y, t) — high density resists price movement.
        # Pressure gradient dp/dy drives order flow just as in Navier-Stokes.
        # ===================================================================
        p = vol_norm.copy()

        # --- Normalize all channels to [-1, 1] via robust σ-clipping ---
        # Financial data is heavy-tailed; 3σ clipping matches fluid simulation statistics
        # where fields also have intermittent large values from coherent structures.
        u = _sigma_normalize(u, self.sigma_clip)
        v = _sigma_normalize(v, self.sigma_clip)
        p = _sigma_normalize(p, self.sigma_clip)

        heatmap = np.stack([u, v, p], axis=0).astype(np.float32)  # (3, H, T)

        # Resize (H=2k, T) → (img_size, img_size)
        t = torch.from_numpy(heatmap).unsqueeze(0)          # (1, 3, H, T)
        t = F.interpolate(
            t, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)                                          # (3, img_size, img_size)

        return t

    def vorticity(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute vorticity w = du/dy - dv/dx from an encoded heatmap.

        w > 0: clockwise rotation (ask pressure driving bid response).
        w < 0: counter-clockwise (bid-side leading).
        Use this to visualize which regions the physics ViT attends to
        and verify that attention heads align with high-vorticity zones.

        Args:
            tensor: (3, H, W) encoded heatmap (output of encode / encode_matrix)
        Returns:
            (H, W) vorticity field
        """
        u = tensor[0]   # price advection
        v = tensor[1]   # volume flux

        du_dy = torch.zeros_like(u)
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / 2.0
        du_dy[0, :]    = u[1, :] - u[0, :]
        du_dy[-1, :]   = u[-1, :] - u[-2, :]

        dv_dx = torch.zeros_like(v)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / 2.0
        dv_dx[:, 0]    = v[:, 1] - v[:, 0]
        dv_dx[:, -1]   = v[:, -1] - v[:, -2]

        return du_dy - dv_dx

    def mid_prices_from_matrix(self, lob_matrix: np.ndarray) -> np.ndarray:
        """Return mid-price series from raw LOB matrix."""
        return (lob_matrix[:, 0] + lob_matrix[:, 2]) / 2.0


def _sigma_normalize(field: np.ndarray, n_sigma: float) -> np.ndarray:
    """
    Clip to ±n*σ around mean, then rescale to [-1, 1].
    Matches the statistical range of NavierStokes velocity/pressure fields
    as stored in PDEArena HDF5 files.
    """
    mu    = field.mean()
    sigma = field.std()
    if sigma < 1e-12:
        return np.zeros_like(field)
    lo = mu - n_sigma * sigma
    hi = mu + n_sigma * sigma
    clipped = np.clip(field, lo, hi)
    span = hi - lo
    return 2.0 * (clipped - lo) / span - 1.0   # [-1, 1]
