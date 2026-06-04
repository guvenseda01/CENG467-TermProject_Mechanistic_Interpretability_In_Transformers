"""
attention_analysis.py
---------------------
Utilities for loading a transformer model via TransformerLens,
extracting attention patterns, and classifying attention head roles.

Usage:
    from src.attention_analysis import load_model, get_attention_patterns, classify_heads
"""

from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Optional
from transformer_lens import HookedTransformer, utils


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str = "gpt2", device: Optional[str] = None) -> HookedTransformer:
    """
    Load a HookedTransformer model for interpretability analysis.

    Args:
        model_name: HuggingFace / TransformerLens model identifier.
        device: 'cpu' or 'cuda'. Auto-detected if None.

    Returns:
        A HookedTransformer model in eval mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    print(f"Loaded '{model_name}' on {device}  |  "
          f"{model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
          f"d_model={model.cfg.d_model}")
    return model


# ---------------------------------------------------------------------------
# Attention pattern extraction
# ---------------------------------------------------------------------------

def get_attention_patterns(
    model: HookedTransformer,
    prompt: str,
    layer: Optional[int] = None,
) -> dict:
    """
    Extract attention patterns for a given prompt.

    Args:
        model: A HookedTransformer instance.
        prompt: Input text.
        layer: If specified, return patterns for that layer only.
                If None, return all layers.

    Returns:
        Dictionary with keys:
            - 'tokens'       : list of token strings
            - 'attention'    : tensor of shape (n_layers, n_heads, seq, seq)  [or (n_heads, seq, seq) if layer given]
            - 'logits'       : final logit tensor
    """
    tokens = model.to_tokens(prompt)
    token_strs = model.to_str_tokens(prompt)

    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    if layer is not None:
        attn = cache[f"blocks.{layer}.attn.hook_pattern"]  # (n_heads, seq, seq)
        return {"tokens": token_strs, "attention": attn, "logits": logits}

    all_attn = torch.stack(
        [cache[f"blocks.{l}.attn.hook_pattern"] for l in range(model.cfg.n_layers)],
        dim=0,
    )  # (n_layers, n_heads, seq, seq)

    return {"tokens": token_strs, "attention": all_attn, "logits": logits}


# ---------------------------------------------------------------------------
# Head classification
# ---------------------------------------------------------------------------

def classify_heads(
    model: HookedTransformer,
    prompts: list[str],
    threshold: float = 0.4,
) -> dict[str, list[tuple[int, int]]]:
    """
    Classify attention heads into functional roles based on their patterns.

    Roles detected:
        - induction       : attends to the token following a previous occurrence of the current token
        - duplicate_token : attends strongly to copies of the current token
        - prev_token      : attends predominantly to the immediately preceding token

    Args:
        model: A HookedTransformer instance.
        prompts: List of example prompts to evaluate heads over.
        threshold: Minimum average attention weight to qualify for a role.

    Returns:
        Dictionary mapping role name → list of (layer, head) tuples.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    induction_scores     = torch.zeros(n_layers, n_heads)
    duplicate_scores     = torch.zeros(n_layers, n_heads)
    prev_token_scores    = torch.zeros(n_layers, n_heads)

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        seq_len = tokens.shape[1]

        for l in range(n_layers):
            attn = cache[f"blocks.{l}.attn.hook_pattern"]  # (n_heads, seq, seq)

            # --- Induction score: attn[h, i, i-1] where tokens[i] == tokens[i-n] ---
            # Simplified: average attention on position (i, i-1) offset for repeated bigrams
            if seq_len > 2:
                induction_scores[l] += attn[:, 1:, :-1].diagonal(dim1=-2, dim2=-1).mean(-1)

            # --- Prev-token score: attention on the immediately preceding token ---
            if seq_len > 1:
                prev_token_scores[l] += attn[:, 1:, :-1].diagonal(dim1=-2, dim2=-1).mean(-1)

            # --- Duplicate-token score: attention on same-token positions ---
            token_ids = tokens[0]  # (seq,)
            dup_mask = (token_ids.unsqueeze(0) == token_ids.unsqueeze(1)).float()  # (seq, seq)
            dup_mask.fill_diagonal_(0)
            dup_attn = (attn * dup_mask.unsqueeze(0)).sum(-1).mean(-1)  # (n_heads,)
            duplicate_scores[l] += dup_attn

    n = len(prompts)
    induction_scores  /= n
    duplicate_scores  /= n
    prev_token_scores /= n

    def _top_heads(score_matrix: torch.Tensor, thresh: float):
        heads = []
        for l in range(n_layers):
            for h in range(n_heads):
                if score_matrix[l, h].item() > thresh:
                    heads.append((l, h))
        return heads

    return {
        "induction":       _top_heads(induction_scores,    threshold),
        "duplicate_token": _top_heads(duplicate_scores,    threshold),
        "prev_token":      _top_heads(prev_token_scores,   threshold),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_attention_head(
    attention: torch.Tensor,
    tokens: list[str],
    layer: int,
    head: int,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot a single attention head's pattern as a heatmap.

    Args:
        attention: Tensor of shape (n_layers, n_heads, seq, seq) or (n_heads, seq, seq).
        tokens: List of token strings.
        layer: Layer index (used when attention has 4 dims).
        head: Head index.
        ax: Matplotlib axes to draw on. Created if None.
        title: Optional plot title.

    Returns:
        The matplotlib Axes object.
    """
    if attention.dim() == 4:
        attn_data = attention[layer, head].cpu().numpy()
    else:
        attn_data = attention[head].cpu().numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        attn_data,
        xticklabels=tokens,
        yticklabels=tokens,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_xlabel("Key (attended to)", fontsize=10)
    ax.set_ylabel("Query (attending from)", fontsize=10)
    ax.set_title(title or f"Layer {layer} · Head {head}", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    return ax


def plot_all_heads(
    attention: torch.Tensor,
    tokens: list[str],
    layer: int,
    figsize_per_head: tuple[int, int] = (4, 3),
) -> plt.Figure:
    """
    Plot all attention heads for a given layer in a grid.

    Args:
        attention: Tensor (n_layers, n_heads, seq, seq).
        tokens: List of token strings.
        layer: Which layer to visualize.
        figsize_per_head: (width, height) per subplot cell.

    Returns:
        The matplotlib Figure.
    """
    n_heads = attention.shape[1]
    ncols = min(4, n_heads)
    nrows = (n_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_head[0] * ncols, figsize_per_head[1] * nrows),
    )
    axes = np.array(axes).flatten()

    for h in range(n_heads):
        plot_attention_head(attention, tokens, layer, h, ax=axes[h])

    for idx in range(n_heads, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"All Attention Heads — Layer {layer}", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
