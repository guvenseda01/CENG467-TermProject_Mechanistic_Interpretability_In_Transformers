"""
visualization.py
----------------
Shared plotting helpers for the CENG467 Group 7 project.

Covers:
    - Ablation comparison bar charts
    - Probability drop distribution histograms
    - Scatter: subject attention score vs probability drop
    - Relation type bar charts
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Ablation comparison
# ---------------------------------------------------------------------------

def plot_ablation_comparison(
    comp_summary: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar charts comparing mean probability drop and prediction change rate
    across subject / random / low-attention ablation types.

    Args:
        comp_summary: DataFrame with columns:
                      ablation_type, mean_prob_drop, change_rate.
        save_path:    If given, save the figure here.

    Returns:
        The matplotlib Figure.
    """
    colors = [
        "#4C72B0" if "subject" in t else
        "#8C8C8C" if "random"  in t else
        "#DD8452"
        for t in comp_summary["ablation_type"].astype(str)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(comp_summary["ablation_type"].astype(str),
                comp_summary["mean_prob_drop"], color=colors)
    axes[0].set_ylabel("Mean Target Probability Drop")
    axes[0].set_title("Subject-Focused vs Random vs Low-Attention Ablation")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(comp_summary["ablation_type"].astype(str),
                comp_summary["change_rate"], color=colors)
    axes[1].set_ylabel("Prediction Change Rate")
    axes[1].set_title("Prediction Change Rate Comparison")
    axes[1].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Probability drop distributions
# ---------------------------------------------------------------------------

def plot_prob_drop_distributions(
    subject_df: pd.DataFrame,
    random_df: pd.DataFrame,
    low_df: pd.DataFrame,
    k_values: list[int] = [5, 10],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram of probability drops for subject / random / low-attention ablations.

    Args:
        subject_df: Top-k ablation DataFrame (has 'top_k' and 'probability_drop').
        random_df:  Random ablation DataFrame (has 'k' and 'probability_drop').
        low_df:     Low-attention ablation DataFrame (has 'k' and 'probability_drop').
        k_values:   List of k values to plot (one subplot per k).
        save_path:  If given, save the figure here.

    Returns:
        The matplotlib Figure.
    """
    fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 4))
    if len(k_values) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        ax.hist(subject_df[subject_df["top_k"] == k]["probability_drop"],
                bins=25, alpha=0.5, label="Subject",   color="#4C72B0")
        ax.hist(random_df[random_df["k"] == k]["probability_drop"],
                bins=25, alpha=0.5, label="Random",    color="#8C8C8C")
        ax.hist(low_df[low_df["k"] == k]["probability_drop"],
                bins=25, alpha=0.5, label="Low-attn",  color="#DD8452")
        ax.set_xlabel("Probability Drop")
        ax.set_ylabel("Count")
        ax.set_title(f"Probability Drop Distribution (Top-{k})")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Scatter: subject attention vs probability drop
# ---------------------------------------------------------------------------

def plot_attention_vs_probdrop(
    scatter_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of subject attention score vs probability drop (single-head ablation).

    Args:
        scatter_df: DataFrame with columns:
                    subject_attention_score, probability_drop, gpt2_correct.
        save_path:  If given, save the figure here.

    Returns:
        The matplotlib Figure.
    """
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(scatter_df["subject_attention_score"],
                    scatter_df["probability_drop"],
                    alpha=0.15, s=6, color="#4C72B0")
    r, p = scipy_stats.pearsonr(scatter_df["subject_attention_score"],
                                 scatter_df["probability_drop"])
    axes[0].set_xlabel("Subject Attention Score")
    axes[0].set_ylabel("Probability Drop after Single-Head Ablation")
    axes[0].set_title(f"All Examples\nPearson r={r:.3f}, p={p:.4f}")

    correct = scatter_df[scatter_df["gpt2_correct"]]
    axes[1].scatter(correct["subject_attention_score"],
                    correct["probability_drop"],
                    alpha=0.2, s=6, color="#2ca02c")
    if len(correct) > 1:
        r2, p2 = scipy_stats.pearsonr(correct["subject_attention_score"],
                                       correct["probability_drop"])
        axes[1].set_title(f"GPT-2 Correct Examples Only\nPearson r={r2:.3f}, p={p2:.4f}")
    axes[1].set_xlabel("Subject Attention Score")
    axes[1].set_ylabel("Probability Drop after Single-Head Ablation")

    plt.suptitle("Subject Attention Score vs Probability Drop (Single-Head Ablation)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Relation type analysis
# ---------------------------------------------------------------------------

def plot_relation_type_analysis(
    relation_summary: pd.DataFrame,
    overall_change_rate: float,
    overall_prob_drop: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar charts of change rate and probability drop per relation type.

    Args:
        relation_summary:    DataFrame with columns: relation_id, change_rate, mean_prob_drop.
        overall_change_rate: Overall mean change rate (drawn as a red dashed reference line).
        overall_prob_drop:   Overall mean probability drop (reference line).
        save_path:           If given, save the figure here.

    Returns:
        The matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(relation_summary["relation_id"],
                 relation_summary["change_rate"], color="#4C72B0")
    axes[0].set_xlabel("Change Rate")
    axes[0].set_title("Prediction Change Rate by Relation Type\n(Top-5 Subject Head Ablation)")
    axes[0].axvline(overall_change_rate, color="red", linestyle="--", label="Overall mean")
    axes[0].legend()

    axes[1].barh(relation_summary["relation_id"],
                 relation_summary["mean_prob_drop"], color="#DD8452")
    axes[1].set_xlabel("Mean Probability Drop")
    axes[1].set_title("Target Probability Drop by Relation Type\n(Top-5 Subject Head Ablation)")
    axes[1].axvline(overall_prob_drop, color="red", linestyle="--", label="Overall mean")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
