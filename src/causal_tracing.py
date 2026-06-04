"""
causal_tracing.py
-----------------
Head-level and residual stream activation patching for causal tracing
experiments (CENG467 Group 7).

Implements the recovery-score metric from Meng et al. (2022):
    recovery = (patched_prob - corrupted_prob) / (clean_prob - corrupted_prob)

Two patching strategies:
    1. head_activation_patching   — patches a single head's hook_z output
    2. residual_position_patching — patches hook_resid_pre at a (layer, pos)

Usage:
    from src.causal_tracing import head_activation_patching, residual_position_patching
"""

from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Shared probability helper
# ---------------------------------------------------------------------------

def _target_prob_from_logits(
    model: HookedTransformer,
    logits: torch.Tensor,
    target: str,
) -> tuple[float, float]:
    """
    Return (probability, logit) of the first subtoken of *target* at the last position.

    Args:
        model:  HookedTransformer instance.
        logits: Tensor of shape (1, seq, vocab).
        target: Target word (space prefix added automatically).

    Returns:
        Tuple of (probability, logit_value).
    """
    next_logits = logits[0, -1]
    probs = torch.softmax(next_logits, dim=-1)
    tid = model.to_tokens(" " + target, prepend_bos=False)[0][0].item()
    return probs[tid].item(), next_logits[tid].item()


# ---------------------------------------------------------------------------
# Head-level activation patching
# ---------------------------------------------------------------------------

def head_activation_patching(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    target: str,
    layer: int,
    head: int,
) -> dict | None:
    """
    Patch a single head's hook_z from the clean run into the corrupted run.

    Args:
        model:             HookedTransformer instance.
        clean_prompt:      Factual prompt (e.g. "The mother tongue of Victor Hugo is").
        corrupted_prompt:  Counterfactual prompt (e.g. "The mother tongue of Albert Einstein is").
        target:            Expected factual token (e.g. "French").
        layer:             Layer index.
        head:              Head index.

    Returns:
        Dict with clean_prob, corrupted_prob, patched_prob, recovery_score,
        or None if prompts tokenise to different lengths.
    """
    clean_tokens     = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    if clean_tokens.shape != corrupted_tokens.shape:
        return None

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    clean_z = clean_cache[f"blocks.{layer}.attn.hook_z"].clone()

    corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

    def patch_fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:, :, head, :] = clean_z[:, :, head, :]
        return value

    patched_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", patch_fn)],
    )

    cp,  _ = _target_prob_from_logits(model, clean_logits,     target)
    crp, _ = _target_prob_from_logits(model, corrupted_logits, target)
    pp,  _ = _target_prob_from_logits(model, patched_logits,   target)

    denom    = cp - crp
    recovery = (pp - crp) / denom if abs(denom) > 1e-6 else 0.0

    return {
        "layer": layer, "head": head,
        "clean_prob":     cp,
        "corrupted_prob": crp,
        "patched_prob":   pp,
        "recovery_score": recovery,
    }


# ---------------------------------------------------------------------------
# Residual stream position patching
# ---------------------------------------------------------------------------

def residual_position_patching(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    target: str,
    layer: int,
    pos: int,
) -> dict | None:
    """
    Patch hook_resid_pre at (layer, pos) from the clean run into the corrupted run.

    This is the standard causal tracing procedure from Meng et al. (2022).

    Args:
        model:             HookedTransformer instance.
        clean_prompt:      Factual prompt.
        corrupted_prompt:  Counterfactual prompt.
        target:            Expected factual token.
        layer:             Layer index.
        pos:               Token position index.

    Returns:
        Dict with layer, position, clean_prob, corrupted_prob, patched_prob,
        recovery_score, or None on token length mismatch.
    """
    clean_tokens     = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    if clean_tokens.shape != corrupted_tokens.shape:
        return None

    hook_name = f"blocks.{layer}.hook_resid_pre"

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits          = model(corrupted_tokens)
    clean_resid               = clean_cache[hook_name].clone()

    def patch_fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:, pos, :] = clean_resid[:, pos, :]
        return value

    patched_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, patch_fn)],
    )

    cp,  _ = _target_prob_from_logits(model, clean_logits,     target)
    crp, _ = _target_prob_from_logits(model, corrupted_logits, target)
    pp,  _ = _target_prob_from_logits(model, patched_logits,   target)

    denom    = cp - crp
    recovery = (pp - crp) / denom if abs(denom) > 1e-6 else 0.0

    return {
        "layer":          layer,
        "position":       pos,
        "clean_prob":     cp,
        "corrupted_prob": crp,
        "patched_prob":   pp,
        "recovery_score": recovery,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_recovery_heatmap(
    recovery_matrix: np.ndarray,
    xlabel: str = "Token Position",
    ylabel: str = "Layer",
    xticklabels: list[str] | None = None,
    title: str = "Activation Patching Recovery Scores",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a recovery score heatmap (layers × positions or layers × heads).

    Args:
        recovery_matrix: 2-D numpy array of recovery scores.
        xlabel:          X-axis label.
        ylabel:          Y-axis label.
        xticklabels:     Optional list of tick labels for the x-axis.
        title:           Plot title.
        save_path:       If given, save the figure here.

    Returns:
        The matplotlib Figure.
    """
    n_rows, n_cols = recovery_matrix.shape

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.7), max(5, n_rows * 0.45)))
    im = ax.imshow(recovery_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recovery Score  (0 = corrupted, 1 = clean)")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"L{l}" for l in range(n_rows)], fontsize=8)

    if xticklabels:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticks(range(n_cols))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
