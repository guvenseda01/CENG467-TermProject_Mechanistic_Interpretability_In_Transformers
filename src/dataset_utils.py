"""
dataset_utils.py
----------------
CounterFact dataset loading and preprocessing for mechanistic
interpretability experiments (CENG467 Group 7).

Usage:
    from src.dataset_utils import load_counterfact, get_sample
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset


NUM_SAMPLES  = 100
RANDOM_SEED  = 42


@dataclass
class CounterFactSample:
    """One record from the CounterFact dataset."""
    sample_id:        int
    subject:          str
    relation_id:      str
    prompt:           str
    target_true:      str
    target_new:       str


def load_counterfact(
    hf_name: str = "azhx/counterfact",
    split: str = "train",
    num_samples: int = NUM_SAMPLES,
    seed: int = RANDOM_SEED,
) -> list[CounterFactSample]:
    """
    Load and shuffle a fixed subset of CounterFact.

    Args:
        hf_name:     HuggingFace dataset identifier.
        split:       Dataset split to use.
        num_samples: Number of samples to select after shuffling.
        seed:        Random seed for reproducibility.

    Returns:
        List of CounterFactSample objects.
    """
    ds = load_dataset(hf_name, split=split)
    ds = ds.shuffle(seed=seed).select(range(num_samples))

    samples = []
    for i, row in enumerate(ds):
        rw = row["requested_rewrite"]
        samples.append(CounterFactSample(
            sample_id   = i,
            subject     = rw["subject"],
            relation_id = rw.get("relation_id", "unknown"),
            prompt      = rw["prompt"].format(rw["subject"]),
            target_true = rw["target_true"]["str"],
            target_new  = rw["target_new"]["str"],
        ))

    print(f"Loaded {len(samples)} CounterFact samples (seed={seed}).")
    return samples


def get_sample(dataset_subset, index: int = 0) -> dict:
    """
    Extract a single sample dict from a HuggingFace dataset subset.
    Mirrors the get_sample() helper used in the notebook.

    Args:
        dataset_subset: A HuggingFace Dataset slice.
        index:          Index within the subset.

    Returns:
        Dict with keys: subject, prompt, target_true, target_new, relation_id.
    """
    sample  = dataset_subset[index]
    rewrite = sample["requested_rewrite"]
    subject = rewrite["subject"]
    return {
        "subject":     subject,
        "prompt":      rewrite["prompt"].format(subject),
        "target_true": rewrite["target_true"]["str"],
        "target_new":  rewrite["target_new"]["str"],
        "relation_id": rewrite.get("relation_id", "unknown"),
    }
