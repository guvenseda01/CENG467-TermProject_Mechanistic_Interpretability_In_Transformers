"""
attention_analysis.py
---------------------
Attention head extraction, subject-token ranking, and ablation helpers
for GPT-2 mechanistic interpretability (CENG467 Group 7).

All functions expect a TransformerLens HookedTransformer model.

Usage:
    from src.attention_analysis import (
        get_subject_token_indices,
        find_top_subject_heads,
        find_low_subject_heads,
        ablate_heads_and_get_logits,
        plot_subject_attention_heatmap,
    )
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Subject token localisation
# ---------------------------------------------------------------------------

def get_subject_token_indices(
    model: HookedTransformer,
    prompt: str,
    subject: str,
) -> list[int]:
    """
    Return the token positions of *subject* inside *prompt*.

    Tries both space-prefixed and bare tokenisations; falls back to
    substring matching if neither produces an exact span match.

    Args:
        model:   HookedTransformer instance.
        prompt:  Full prompt string.
        subject: Entity string to locate.

    Returns:
        Sorted list of token indices (may be empty if not found).
    """
    prompt_tokens = model.to_str_tokens(prompt)

    candidates = [
        model.to_str_tokens(subject,        prepend_bos=False),
        model.to_str_tokens(" " + subject,  prepend_bos=False),
    ]

    for subject_tokens in candidates:
        subject_clean = [t for t in subject_tokens if t != "<|endoftext|>"]
        indices: list[int] = []
        for start in range(len(prompt_tokens)):
            span = prompt_tokens[start: start + len(subject_clean)]
            if len(span) != len(subject_clean):
                continue
            if [t.strip() for t in span] == [t.strip() for t in subject_clean]:
                indices.extend(range(start, start + len(subject_clean)))
        if indices:
            return sorted(set(indices))

    # Fallback: substring match
    subject_tokens_fb = [
        t for t in model.to_str_tokens(subject, prepend_bos=False)
        if t != "<|endoftext|>"
    ]
    indices = []
    for i, token in enumerate(prompt_tokens):
        if token == "<|endoftext|>":
            continue
        for st in subject_tokens_fb:
            if st.strip() and st.strip() in token.strip():
                indices.append(i)
    return sorted(set(indices))


# ---------------------------------------------------------------------------
# Head ranking
# ---------------------------------------------------------------------------

def _rank_heads_by_subject(
    model: HookedTransformer,
    prompt: str,
    subject: str,
) -> pd.DataFrame | None:
    """
    Score every (layer, head) by the sum of final-query attention to subject tokens.

    Returns a DataFrame sorted by score descending, or None if subject not found.
    """
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)

    subj_idx = get_subject_token_indices(model, prompt, subject)
    if not subj_idx:
        return None

    rows = []
    for layer in range(model.cfg.n_layers):
        attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]   # (n_heads, seq, seq)
        for head in range(model.cfg.n_heads):
            score = attn[head][-1].detach().cpu()[subj_idx].sum().item()
            rows.append({"layer": layer, "head": head, "score": score})

    return pd.DataFrame(rows).sort_values("score", ascending=False)


def find_top_subject_heads(
    model: HookedTransformer,
    prompt: str,
    subject: str,
    top_k: int = 5,
) -> list[tuple[int, int]]:
    """Return the top-k (layer, head) pairs with the highest subject attention."""
    df = _rank_heads_by_subject(model, prompt, subject)
    if df is None:
        return []
    top = df.head(top_k)
    return [(int(top.iloc[i]["layer"]), int(top.iloc[i]["head"])) for i in range(len(top))]


def find_low_subject_heads(
    model: HookedTransformer,
    prompt: str,
    subject: str,
    top_k: int = 5,
) -> list[tuple[int, int]]:
    """Return the top-k (layer, head) pairs with the *lowest* subject attention (control)."""
    df = _rank_heads_by_subject(model, prompt, subject)
    if df is None:
        return []
    low = df.sort_values("score").head(top_k)
    return [(int(low.iloc[i]["layer"]), int(low.iloc[i]["head"])) for i in range(len(low))]


def get_random_heads(
    model: HookedTransformer,
    k: int,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Return k randomly sampled (layer, head) pairs as a control group."""
    rng = random.Random(seed)
    all_heads = [
        (l, h)
        for l in range(model.cfg.n_layers)
        for h in range(model.cfg.n_heads)
    ]
    return rng.sample(all_heads, k)


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def ablate_heads_and_get_logits(
    model: HookedTransformer,
    prompt: str,
    heads_to_ablate: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Zero-ablate the listed (layer, head) pairs via hook_z and return logits.

    Multiple heads in the same layer are handled in a single hook call to
    avoid hook-registration conflicts.

    Args:
        model:           HookedTransformer instance.
        prompt:          Input prompt string.
        heads_to_ablate: List of (layer_idx, head_idx) pairs.

    Returns:
        Logit tensor of shape (1, seq_len, vocab_size).
    """
    tokens = model.to_tokens(prompt)

    layer_to_heads: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads_to_ablate:
        layer_to_heads[layer].append(head)

    def make_hook(heads_in_layer: list[int]):
        def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
            # value: (batch, seq_len, n_heads, d_head)
            for h in heads_in_layer:
                value[:, :, h, :] = 0
            return value
        return hook_fn

    fwd_hooks = [
        (f"blocks.{layer}.attn.hook_z", make_hook(heads))
        for layer, heads in layer_to_heads.items()
    ]
    return model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_subject_attention_heatmap(
    model: HookedTransformer,
    prompt: str,
    subject: str,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a (layer × head) heatmap of subject attention scores.

    Args:
        model:     HookedTransformer instance.
        prompt:    Input prompt string.
        subject:   Subject entity string.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    subj_idx = get_subject_token_indices(model, prompt, subject)

    heat = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(model.cfg.n_layers):
        attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]
        for head in range(model.cfg.n_heads):
            if subj_idx:
                heat[layer, head] = attn[head][-1].detach().cpu()[subj_idx].sum().item()

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(heat, aspect="auto", cmap="Blues")
    plt.colorbar(im, ax=ax, label="Subject Attention Score (final query → subject tokens)")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f'Subject Attention Heatmap\nPrompt: "{prompt}"')
    ax.set_xticks(range(model.cfg.n_heads))
    ax.set_yticks(range(model.cfg.n_layers))

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
