"""
Visualization utilities for the Physics2Finance project.

Generates publication-quality figures for:
- LOB heatmap samples vs. fluid field snapshots (visual analogy)
- Structure function scaling exponents (turbulence validation)
- Latent embedding t-SNE/UMAP projections
- Forecast performance comparison across models and horizons
- Diebold-Mariano test result tables
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger


def plot_fluid_vs_lob(
    fluid_field: np.ndarray,  # (3, H, W) RGB fluid image
    lob_heatmap: np.ndarray,  # (3, H, W) RGB LOB heatmap
    save_path: Optional[str] = None,
    titles: Tuple[str, str] = ("Turbulent Fluid (DNS)", "Limit Order Book"),
) -> plt.Figure:
    """
    Side-by-side comparison of a fluid field snapshot and LOB heatmap,
    with individual channel breakdown.
    """
    channel_names = {0: "R: Ask / Pressure", 1: "G: Bid / Density", 2: "B: Imbalance / Velocity"}

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.2)

    for col, (data, title) in enumerate([(fluid_field, titles[0]), (lob_heatmap, titles[1])]):
        # RGB composite
        ax = fig.add_subplot(gs[0, col * 2: col * 2 + 2])
        rgb = np.transpose(data, (1, 2, 0))
        rgb = np.clip(rgb, 0, 1)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

        # Individual channels
        for ch in range(3):
            ax_ch = fig.add_subplot(gs[1, col * 2 + (ch if ch < 2 else 0)])
            cmap = ["Reds", "Greens", "Blues"][ch]
            im = ax_ch.imshow(data[ch], cmap=cmap, aspect="auto")
            ax_ch.set_title(channel_names[ch], fontsize=8)
            ax_ch.axis("off")
            plt.colorbar(im, ax=ax_ch, fraction=0.046, pad=0.04)

    fig.suptitle("Turbulence ↔ Market Microstructure: RGB Analogy", fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_structure_functions(
    r_vals: np.ndarray,
    sq_vals: np.ndarray,
    orders: List[int] = (2, 4, 6),
    k41_zeta: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Log-log plot of velocity structure functions S_q(r) vs. separation r.
    Deviations from K41 scaling (q/3) indicate intermittency.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))
    for i, (q, sq) in enumerate(zip(orders, sq_vals)):
        ax.loglog(r_vals, sq, "o-", color=colors[i], label=f"q={q}", markersize=3)
    ax.set_xlabel("Separation r", fontsize=11)
    ax.set_ylabel("$S_q(r)$", fontsize=11)
    ax.set_title("Velocity Structure Functions", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    fitted_zeta = []
    for q, sq in zip(orders, sq_vals):
        log_r = np.log(r_vals[r_vals > 1])
        log_sq = np.log(sq[r_vals > 1] + 1e-30)
        coef = np.polyfit(log_r, log_sq, 1)
        fitted_zeta.append(coef[0])

    ax2.plot(orders, fitted_zeta, "bo-", label="Observed ζ(q)", linewidth=2)
    if k41_zeta is not None:
        ax2.plot(orders, k41_zeta, "r--", label="K41: ζ(q)=q/3", linewidth=2)
    else:
        k41 = [q / 3 for q in orders]
        ax2.plot(orders, k41, "r--", label="K41: ζ(q)=q/3", linewidth=2)

    ax2.set_xlabel("Order q", fontsize=11)
    ax2.set_ylabel("Scaling exponent ζ(q)", fontsize=11)
    ax2.set_title("Multifractal Scaling Exponents", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_forecast_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    horizon: int,
    n_plot: int = 500,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Time-series plot comparing multiple model volatility forecasts vs. realized RV.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    t = np.arange(min(n_plot, len(y_true)))
    y = y_true[:len(t)]

    ax = axes[0]
    ax.plot(t, y, "k-", label="Realized Volatility", linewidth=1, alpha=0.8)
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    for (name, preds), color in zip(predictions.items(), colors):
        p = np.asarray(preds)[:len(t)]
        ax.plot(t, p, "-", label=name, linewidth=1, alpha=0.7, color=color)
    ax.set_ylabel("Realized Volatility", fontsize=11)
    ax.set_title(f"Volatility Forecasts (horizon={horizon} ticks)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for (name, preds), color in zip(predictions.items(), colors):
        p = np.asarray(preds)[:len(t)]
        error = p - y
        ax2.plot(t, error, "-", label=name, linewidth=0.8, alpha=0.7, color=color)
    ax2.axhline(0, color="k", linestyle="--", linewidth=1)
    ax2.set_xlabel("Time (ticks)", fontsize=11)
    ax2.set_ylabel("Forecast Error", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_table(
    results: Dict[str, Dict[str, float]],
    horizons: List[int],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap table of MAE/RMSE/QLIKE metrics across models and horizons.
    """
    metrics = ["mae", "rmse", "qlike"]
    model_names = list(results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), max(3, n_models)))

    for ax, metric in zip(axes, metrics):
        matrix = np.array([
            [results[m].get(metric, np.nan) for m in model_names]
        ])

        im = ax.imshow(matrix.T, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks([0])
        ax.set_xticklabels([f"h={horizons[0]}"], fontsize=10)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_title(metric.upper(), fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax)

        for i, m in enumerate(model_names):
            val = results[m].get(metric, np.nan)
            ax.text(0, i, f"{val:.4f}", ha="center", va="center", fontsize=9, fontweight="bold")

    fig.suptitle("Model Comparison: Volatility Forecasting Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "umap",
    n_components: int = 2,
    title: str = "Latent Embedding Projection",
    label_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: float = 30.0,
    max_samples: int = 5000,
) -> plt.Figure:
    """
    2D UMAP or t-SNE projection of backbone embeddings.

    Shows whether physics embeddings cluster LOB states meaningfully — a key
    qualitative result for the paper demonstrating cross-domain structure.

    Args:
        embeddings:  (N, D) backbone embeddings.
        labels:      (N,) optional color labels (e.g., RV quantile bins).
        method:      "umap" or "tsne".
        label_names: map from integer label to human-readable string.
        max_samples: subsample if N > this for speed.
    """
    if len(embeddings) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        if labels is not None:
            labels = labels[idx]
        logger.info(f"Subsampled to {max_samples} for {method}")

    method = method.lower()
    if method == "umap":
        try:
            import umap as umap_lib
        except ImportError:
            raise ImportError("umap-learn not installed: pip install umap-learn")
        reducer = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            verbose=False,
        )
        proj = reducer.fit_transform(embeddings)
        method_label = "UMAP"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        proj = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_jobs=-1,
        ).fit_transform(embeddings)
        method_label = "t-SNE"
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'umap' or 'tsne'.")

    logger.info(f"{method_label}: {embeddings.shape} → {proj.shape}")

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap("viridis", len(unique_labels))
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            name = (label_names or {}).get(int(lbl), str(lbl))
            ax.scatter(proj[mask, 0], proj[mask, 1], c=[cmap(i)],
                       label=name, s=4, alpha=0.6, rasterized=True)
        ax.legend(markerscale=3, fontsize=9, title="Label", title_fontsize=10)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=3, alpha=0.4,
                   c="steelblue", rasterized=True)

    ax.set_xlabel(f"{method_label} dim 1", fontsize=12)
    ax.set_ylabel(f"{method_label} dim 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved embedding projection → {save_path}")
    return fig


def make_rv_quantile_labels(
    rv_targets: np.ndarray,
    n_quantiles: int = 5,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Bin RV values into quantile labels for coloring embedding projections."""
    quantiles = np.quantile(rv_targets, np.linspace(0, 1, n_quantiles + 1))
    labels = np.digitize(rv_targets, quantiles[1:-1])
    label_names = {
        i: f"RV Q{i+1} ({quantiles[i]:.2e}–{quantiles[i+1]:.2e})"
        for i in range(n_quantiles)
    }
    return labels.astype(np.int32), label_names
