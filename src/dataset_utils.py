"""
dataset_utils.py
----------------
Utilities for loading and preprocessing the CounterFact and SQuAD datasets
for use in mechanistic interpretability experiments.

Usage:
    from src.dataset_utils import load_counterfact, load_squad, CounterFactSample
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Data containers
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


@dataclass
class SQuADSample:
    """One record from the SQuAD dataset."""
    id: str
    context: str
    question: str
    answer: str
    answer_start: int


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
# SQuAD
# ---------------------------------------------------------------------------

def load_squad(
    split: str = "validation",
    max_samples: Optional[int] = 500,
    seed: int = 42,
    min_answer_length: int = 1,
) -> list[SQuADSample]:
    """
    Load a subset of the SQuAD v1.1 dataset.

    Args:
        split: 'train' or 'validation'.
        max_samples: Maximum number of samples to return.
        seed: Random seed for sampling.
        min_answer_length: Minimum answer character length to include.

    Returns:
        List of SQuADSample objects.
    """
    ds = load_dataset("rajpurkar/squad", split=split)

    # Filter to single-answer examples with non-trivial answers
    filtered = [
        row for row in ds
        if len(row["answers"]["text"]) > 0
        and len(row["answers"]["text"][0]) >= min_answer_length
    ]

    if max_samples is not None and max_samples < len(filtered):
        random.seed(seed)
        filtered = random.sample(filtered, max_samples)

    samples = []
    for row in filtered:
        answer_text = row["answers"]["text"][0]
        answer_start = row["answers"]["answer_start"][0]
        samples.append(SQuADSample(
            id=row["id"],
            context=row["context"],
            question=row["question"],
            answer=answer_text,
            answer_start=answer_start,
        ))

    print(f"Loaded {len(samples)} SQuAD samples.")
    return samples


# ---------------------------------------------------------------------------
# Prompt builders
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


def squad_to_prompt(sample: SQuADSample, max_context_chars: int = 512) -> tuple[str, str, str]:
    """
    Convert a SQuADSample to (prompt, subject, target) for attention analysis.

    Args:
        sample: A SQuADSample.
        max_context_chars: Maximum context characters to include (for GPU memory).

    Returns:
        Tuple of (prompt_str, question_str, answer_str).
    """
    context = sample.context[:max_context_chars]
    prompt = f"Context: {context}\nQuestion: {sample.question}\nAnswer:"
    return prompt, sample.question, " " + sample.answer
