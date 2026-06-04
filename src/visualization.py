"""
visualization.py
----------------
Shared plotting helpers for the Mechanistic Interpretability project.
Covers summary plots for multi-sample causal tracing, head score matrices,
and attention rollout visualizations.

Usage:
    from src.visualization import plot_aggregate_causal_trace, plot_head_scores
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Optional


# ---------------------------------------------------------------------------
# Causal trace — aggregate over many samples
# ---------------------------------------------------------------------------

def plot_aggregate_causal_trace(
    indirect_effects: list[np.ndarray],
    token_kind: str = "subject last",
    figsize: tuple[int, int] = (10, 5),
    title: Optional[str] = None,
    cmap: str = "Purples",
) -> plt.Figure:
    """
    Average indirect effect heatmaps from multiple causal tracing runs and plot.

    Args:
        indirect_effects: List of (n_layers, seq) indirect effect arrays.
                          All arrays must have the same shape.
        token_kind: Label for x-axis description.
        figsize: Figure size.
        title: Optional title override.
        cmap: Colormap.

    Returns:
        The matplotlib Figure.
    """
    stacked = np.stack(indirect_effects, axis=0)      # (N, n_layers, seq)
    mean_ie = stacked.mean(axis=0)                     # (n_layers, seq)
    n_layers, seq_len = mean_ie.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mean_ie, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xlabel(f"Token position ({token_kind})", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)
    ax.set_xticks(range(seq_len))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.colorbar(im, ax=ax, label="Mean Indirect Effect", shrink=0.8)
    ax.set_title(title or f"Aggregate Causal Trace (n={len(indirect_effects)})", fontsize=13)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Attention head score matrix
# ---------------------------------------------------------------------------

def plot_head_scores(
    scores: np.ndarray,
    title: str = "Attention Head Scores",
    cmap: str = "YlOrRd",
    annot: bool = True,
    figsize: Optional[tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot a (n_layers × n_heads) score matrix as a labeled heatmap.

    Args:
        scores: 2D numpy array of shape (n_layers, n_heads).
        title: Plot title.
        cmap: Colormap.
        annot: Whether to annotate cells with numeric values.
        figsize: Figure size. Auto-computed if None.

    Returns:
        The matplotlib Figure.
    """
    n_layers, n_heads = scores.shape
    if figsize is None:
        figsize = (max(6, n_heads * 0.8), max(4, n_layers * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        scores,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=scores.max() if scores.max() > 0 else 1,
        annot=annot,
        fmt=".2f",
        xticklabels=[f"H{h}" for h in range(n_heads)],
        yticklabels=[f"L{l}" for l in range(n_layers)],
        linewidths=0.4,
        linecolor="lightgrey",
        cbar_kws={"shrink": 0.7},
    )
    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Token probability bar chart
# ---------------------------------------------------------------------------

def plot_top_predictions(
    logits: "torch.Tensor",   # noqa: F821
    tokenizer,
    top_k: int = 10,
    title: str = "Top-k Predictions",
    figsize: tuple[int, int] = (8, 4),
) -> plt.Figure:
    """
    Bar chart of the top-k predicted tokens at the last position.

    Args:
        logits: Model logits tensor, shape (batch, seq, vocab) or (seq, vocab).
        tokenizer: The model's tokenizer (must have decode()).
        top_k: Number of top tokens to show.
        title: Plot title.
        figsize: Figure size.

    Returns:
        The matplotlib Figure.
    """
    import torch

    if logits.dim() == 3:
        last = logits[0, -1, :]
    else:
        last = logits[-1, :]

    probs = torch.softmax(last, dim=-1)
    top_probs, top_ids = probs.topk(top_k)
    top_probs = top_probs.detach().cpu().numpy()
    top_tokens = [repr(tokenizer.decode([tid.item()])) for tid in top_ids]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(top_k), top_probs[::-1], color="steelblue", edgecolor="white")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_tokens[::-1], fontsize=9)
    ax.set_xlabel("Probability", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, top_probs[0] * 1.15)

    for bar, prob in zip(bars, top_probs[::-1]):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}", va="center", fontsize=8,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------

def compute_attention_rollout(
    attention_matrices: "torch.Tensor",  # (n_layers, n_heads, seq, seq)
    add_residual: bool = True,
) -> np.ndarray:
    """
    Compute attention rollout (Abnar & Zuidema, 2020).

    Rollout recursively multiplies attention matrices from layer 0 to L,
    optionally adding a residual identity term to model skip connections.

    Args:
        attention_matrices: Tensor of shape (n_layers, n_heads, seq, seq).
        add_residual: Whether to add 0.5 * identity to each layer's attention.

    Returns:
        Rollout matrix as numpy array, shape (seq, seq).
    """
    import torch

    n_layers, n_heads, seq, _ = attention_matrices.shape
    # Average over heads
    avg_attn = attention_matrices.mean(dim=1)    # (n_layers, seq, seq)

    rollout = torch.eye(seq, device=attention_matrices.device)
    for l in range(n_layers):
        A = avg_attn[l]
        if add_residual:
            A = 0.5 * A + 0.5 * torch.eye(seq, device=A.device)
        # Normalize rows
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = A @ rollout

    return rollout.cpu().numpy()
