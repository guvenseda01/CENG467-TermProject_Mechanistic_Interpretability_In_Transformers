"""
dataset_utils.py
----------------
Utilities for loading and preprocessing the CounterFact dataset
for use in mechanistic interpretability experiments.

Usage:
    from src.dataset_utils import load_counterfact, CounterFactSample
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CounterFactSample:
    """One record from the CounterFact dataset."""
    case_id: int
    subject: str
    relation: str
    target_true: str        # The correct factual completion
    target_new: str         # The counterfactual completion
    prompt: str             # "{subject} ... is"
    rephrase_prompts: list[str]


# ---------------------------------------------------------------------------
# CounterFact
# ---------------------------------------------------------------------------

def load_counterfact(
    split: str = "train",
    max_samples: Optional[int] = 500,
    seed: int = 42,
) -> list[CounterFactSample]:
    """
    Load a subset of the CounterFact dataset.

    Args:
        split: Dataset split ('train' is the only available split).
        max_samples: Maximum number of samples to return. None for all.
        seed: Random seed for sampling.

    Returns:
        List of CounterFactSample objects.
    """
    ds = load_dataset("NeelNanda/counterfact-tracing", split=split)

    indices = list(range(len(ds)))
    if max_samples is not None and max_samples < len(indices):
        random.seed(seed)
        indices = random.sample(indices, max_samples)

    samples = []
    for i in indices:
        row = ds[i]
        samples.append(CounterFactSample(
            case_id=row.get("case_id", i),
            subject=row["requested_rewrite"]["subject"],
            relation=row["requested_rewrite"]["relation_id"],
            target_true=row["requested_rewrite"]["target_true"]["str"],
            target_new=row["requested_rewrite"]["target_new"]["str"],
            prompt=row["requested_rewrite"]["prompt"].format(
                row["requested_rewrite"]["subject"]
            ),
            rephrase_prompts=row.get("paraphrase_prompts", []),
        ))

    print(f"Loaded {len(samples)} CounterFact samples.")
    return samples


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def counterfact_to_prompt(sample: CounterFactSample, use_rephrase: bool = False) -> tuple[str, str, str]:
    """
    Convert a CounterFactSample to (prompt, subject, target) for causal tracing.

    Args:
        sample: A CounterFactSample.
        use_rephrase: If True and rephrase prompts exist, use a random one.

    Returns:
        Tuple of (prompt_str, subject_str, target_true_str).
    """
    if use_rephrase and sample.rephrase_prompts:
        prompt = random.choice(sample.rephrase_prompts)
    else:
        prompt = sample.prompt
    return prompt, sample.subject, " " + sample.target_true
