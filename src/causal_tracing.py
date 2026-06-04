"""
causal_tracing.py
-----------------
Implements the causal tracing / activation patching methodology
from Meng et al. (2022) "Locating and Editing Factual Associations in GPT".

The core idea:
    1. Run the model on a CLEAN prompt  → cache all activations.
    2. Run the model on a CORRUPTED prompt (subject tokens replaced with noise).
    3. For each (layer, token position), patch the corrupted run with the
       clean activation and measure how much the target token probability recovers.
    4. The resulting "indirect effect" heatmap shows where factual knowledge lives.

Usage:
    from src.causal_tracing import run_causal_trace, plot_causal_trace
"""

from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Callable
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Helper: corrupt subject tokens with gaussian noise
# ---------------------------------------------------------------------------

def _corrupt_tokens(
    embeddings: torch.Tensor,
    subject_positions: list[int],
    noise_scale: float = 3.0,
) -> torch.Tensor:
    """
    Add gaussian noise to the embedding of subject token positions.

    Args:
        embeddings: Token embedding tensor, shape (batch, seq, d_model).
        subject_positions: List of token indices corresponding to the subject.
        noise_scale: Standard deviation of the noise (scaled by embedding std).

    Returns:
        Corrupted embeddings tensor with the same shape.
    """
    corrupted = embeddings.clone()
    std = embeddings.std().item()
    noise = torch.randn_like(embeddings[:, subject_positions, :]) * noise_scale * std
    corrupted[:, subject_positions, :] += noise
    return corrupted


# ---------------------------------------------------------------------------
# Core causal trace
# ---------------------------------------------------------------------------

def run_causal_trace(
    model: HookedTransformer,
    prompt: str,
    subject: str,
    target: str,
    noise_scale: float = 3.0,
    patch_component: str = "resid_post",   # 'resid_post' | 'mlp_post' | 'attn_out'
) -> dict:
    """
    Run a full causal tracing experiment for one (prompt, subject, target) triple.

    Args:
        model: HookedTransformer model.
        prompt: The factual prompt, e.g. "The capital of France is".
        subject: The subject entity string within the prompt, e.g. "France".
        target: The expected factual completion token, e.g. " Paris".
        noise_scale: Noise level for corrupting subject embeddings.
        patch_component: Which activation to patch.
            'resid_post' — full residual stream after each layer (default, matches ROME paper)
            'mlp_post'   — MLP output only
            'attn_out'   — attention output only

    Returns:
        Dictionary with:
            'clean_prob'     : probability of target on clean run
            'corrupt_prob'   : probability of target on corrupted run
            'indirect_effect': 2D numpy array (n_layers, seq_len) of indirect effects
            'tokens'         : list of token strings
            'subject_positions': list of token positions for the subject
    """
    tokenizer = model.tokenizer
    device = next(model.parameters()).device

    # Tokenize
    tokens = model.to_tokens(prompt)                      # (1, seq)
    token_strs = model.to_str_tokens(prompt)
    target_id = model.to_single_token(target)

    # Find subject positions
    subject_tokens = model.to_tokens(subject, prepend_bos=False)[0].tolist()
    subject_positions = _find_subject_positions(tokens[0].tolist(), subject_tokens)

    if not subject_positions:
        raise ValueError(
            f"Subject '{subject}' (tokens {subject_tokens}) not found in prompt tokens."
        )

    # ── 1. Clean run ──────────────────────────────────────────────────────────
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(tokens, remove_batch_dim=False)

    clean_prob = _target_prob(clean_logits, target_id)

    # ── 2. Corrupted run ──────────────────────────────────────────────────────
    def corrupt_hook(value: torch.Tensor, hook) -> torch.Tensor:
        """Hook that corrupts the embedding layer output for subject positions."""
        return _corrupt_tokens(value, subject_positions, noise_scale)

    with torch.no_grad():
        corrupt_logits, corrupt_cache = model.run_with_cache(
            tokens,
            fwd_hooks=[("hook_embed", corrupt_hook)],
            remove_batch_dim=False,
        )

    corrupt_prob = _target_prob(corrupt_logits, target_id)

    # ── 3. Patching sweep ─────────────────────────────────────────────────────
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[1]
    indirect_effect = np.zeros((n_layers, seq_len))

    for layer_idx in range(n_layers):
        for pos in range(seq_len):
            hook_name = _get_hook_name(layer_idx, patch_component)
            clean_act = clean_cache[hook_name]  # (1, seq, d_model)

            def patch_hook(value: torch.Tensor, hook, _l=layer_idx, _p=pos) -> torch.Tensor:
                patched = value.clone()
                patched[:, _p, :] = clean_act[:, _p, :]
                return patched

            with torch.no_grad():
                patched_logits = model.run_with_cache(
                    tokens,
                    fwd_hooks=[
                        ("hook_embed", corrupt_hook),
                        (hook_name, patch_hook),
                    ],
                    remove_batch_dim=False,
                )[0]

            patched_prob = _target_prob(patched_logits, target_id)
            # Indirect effect: how much probability recovered relative to clean–corrupt gap
            gap = clean_prob - corrupt_prob
            if abs(gap) > 1e-8:
                indirect_effect[layer_idx, pos] = (patched_prob - corrupt_prob) / gap
            else:
                indirect_effect[layer_idx, pos] = 0.0

    return {
        "clean_prob": clean_prob,
        "corrupt_prob": corrupt_prob,
        "indirect_effect": indirect_effect,
        "tokens": token_strs,
        "subject_positions": subject_positions,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_causal_trace(
    result: dict,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 6),
    cmap: str = "RdYlGn",
) -> plt.Figure:
    """
    Visualize the indirect effect heatmap from a causal tracing result.

    Args:
        result: Output dict from run_causal_trace().
        title: Optional figure title.
        figsize: Figure size.
        cmap: Matplotlib colormap name.

    Returns:
        The matplotlib Figure.
    """
    ie = result["indirect_effect"]     # (n_layers, seq)
    tokens = result["tokens"]
    subject_positions = result["subject_positions"]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        ie,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-0.1,
        vmax=1.0,
        xticklabels=tokens,
        yticklabels=[f"L{l}" for l in range(ie.shape[0])],
        linewidths=0.3,
        linecolor="grey",
        cbar_kws={"label": "Indirect Effect (normalized)", "shrink": 0.8},
    )

    # Highlight subject token columns
    for pos in subject_positions:
        ax.add_patch(plt.Rectangle(
            (pos, 0), 1, ie.shape[0],
            fill=False, edgecolor="blue", linewidth=2, linestyle="--",
        ))

    ax.set_xlabel("Token Position", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    clean_p = result["clean_prob"]
    corrupt_p = result["corrupt_prob"]
    default_title = (
        f"Causal Trace  |  clean p={clean_p:.3f}  corrupt p={corrupt_p:.3f}\n"
        f"Blue dashed = subject tokens"
    )
    ax.set_title(title or default_title, fontsize=12)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _target_prob(logits: torch.Tensor, target_id: int) -> float:
    """Return the probability of target_id at the final token position."""
    last_logits = logits[0, -1, :]          # (vocab,)
    probs = torch.softmax(last_logits, dim=-1)
    return probs[target_id].item()


def _find_subject_positions(token_ids: list[int], subject_token_ids: list[int]) -> list[int]:
    """Find start positions of subject_token_ids inside token_ids."""
    positions = []
    n = len(subject_token_ids)
    for i in range(len(token_ids) - n + 1):
        if token_ids[i: i + n] == subject_token_ids:
            positions = list(range(i, i + n))
            break
    return positions


def _get_hook_name(layer: int, component: str) -> str:
    """Map a component name to the TransformerLens hook point name."""
    mapping = {
        "resid_post": f"blocks.{layer}.hook_resid_post",
        "mlp_post":   f"blocks.{layer}.hook_mlp_out",
        "attn_out":   f"blocks.{layer}.attn.hook_z",
    }
    if component not in mapping:
        raise ValueError(f"Unknown component '{component}'. Choose from: {list(mapping)}")
    return mapping[component]
