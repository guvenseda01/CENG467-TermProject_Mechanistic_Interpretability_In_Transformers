# Mechanistic Interpretability in Transformers
### CENG467 — Term Project | Group 7

> Analyzing attention head behavior and performing causal tracing experiments to understand how factual knowledge emerges in large language models.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Experiments](#experiments)
- [Outputs](#outputs)
- [References](#references)

---

## Overview

This project investigates the internal mechanisms of GPT-2 through **mechanistic interpretability**. We focus on two core techniques:

| Technique | Goal |
|---|---|
| **Attention Head Analysis** | Identify which heads attend most to subject tokens and measure how ablating them drops factual recall |
| **Causal Tracing** | Localize where factual knowledge is stored by patching head-level and residual stream activations |

The experiments are conducted on the **CounterFact** dataset using 300 randomly sampled records.

---

## Project Structure

```
CENG467-TermProject_Mechanistic_Interpretability_In_Transformers/
│
├── CENG467_Group7_Final_Version.ipynb   # Main experiment notebook
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
└── src/
    ├── attention_analysis.py            # Attention head extraction, classification, ablation
    ├── causal_tracing.py                # Head-level and residual stream activation patching
    ├── dataset_utils.py                 # CounterFact loading and preprocessing
    └── visualization.py                 # Heatmaps, scatter plots, distribution charts
```

Output directories created automatically by the notebook:

```
outputs/
    figures/    ← PNG plots
    tables/     ← CSV result tables
results/        ← Summary CSVs and statistical test outputs
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended

### Installation

```bash
git clone https://github.com/guvenseda01/CENG467-TermProject_Mechanistic_Interpretability_In_Transformers.git
cd CENG467-TermProject_Mechanistic_Interpretability_In_Transformers

pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook CENG467_Group7_Final_Version.ipynb
```

Or open directly in Google Colab (recommended for GPU access):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guvenseda01/CENG467-TermProject_Mechanistic_Interpretability_In_Transformers/blob/main/CENG467_Group7_Final_Version.ipynb)

---

## Dataset

### CounterFact (`azhx/counterfact`)
- Factual statements paired with counterfactual alternatives (e.g. *"The mother tongue of Victor Hugo is French"* → target_new: *"German"*).
- Loaded via HuggingFace `datasets`: `load_dataset("azhx/counterfact")`
- 300 randomly sampled records used (`RANDOM_SEED=42`)
- Samples filtered into **GPT-2 correct** and **GPT-2 incorrect** subsets for separate analysis

---

## Experiments

### 1. Baseline Factual Accuracy

For each sample, the notebook records:
- GPT-2's predicted next token vs. the true target
- Target token probability, logit, and rank
- Whether the target appears in the top-5 predictions

Results saved to `outputs/tables/baseline_stats.csv`.

### 2. Top-k Subject Head Ablation

Heads are ranked by how strongly their final-query attention focuses on the subject entity tokens. The top-k heads are zeroed out (`hook_z`) and the effect on prediction and probability is measured.

`TOP_K_VALUES = [1, 3, 5, 10]`

Metrics recorded: prediction change rate, probability drop, logit drop, rank drop, KL divergence.

Results saved to `outputs/tables/topk_ablation_results.csv`.

### 3. Control Groups

Two control ablations run at k=5 and k=10:
- **Random heads** — k randomly selected (layer, head) pairs
- **Low-attention heads** — k heads with the *lowest* subject attention scores

Allows testing whether subject-focused heads are specifically important, or whether ablating any heads causes similar disruption.

### 4. Relation Type Analysis

Ablation impact broken down by `relation_id` (e.g. country-capital, language, employer) to identify which fact types are most dependent on subject-attending heads.

Plot saved to `outputs/figures/relation_type_analysis.png`.

### 5. Subject Attention vs. Probability Drop (Scatter)

Scatter plot of each head's subject attention score against the probability drop caused by ablating that single head. Pearson correlation computed for all samples and for GPT-2-correct samples separately.

Plot saved to `outputs/figures/scatter_attention_vs_probdrop.png`.

### 6. Head-Level Activation Patching

For three hand-picked (clean, corrupted, target) pairs, each head's output is copied from the clean run into the corrupted run. The **recovery score** measures how much of the target probability is restored:

```
recovery = (patched_prob − corrupted_prob) / (clean_prob − corrupted_prob)
```

Recovery heatmaps saved to `outputs/figures/head_patching_<target>.png`.

### 7. Residual Stream Causal Tracing

For each (layer, token position), the residual stream activation (`hook_resid_pre`) is patched from clean into corrupted. The resulting recovery heatmap is the standard causal tracing figure from Meng et al. (2022).

Heatmaps saved to `outputs/figures/causal_tracing_<target>.png`.

### 8. Statistical Significance

- **Chi-square** tests on prediction change rates (subject vs. random, subject vs. low-attention)
- **Mann-Whitney U** tests on probability drops
- **Wilcoxon signed-rank** tests on paired probability drops
- **Bootstrap 95% confidence intervals** on mean probability drop

Results saved to `results/statistical_tests.csv`.

---

## Outputs

| File | Description |
|---|---|
| `outputs/tables/baseline_stats.csv` | Per-sample baseline prediction and probability stats |
| `outputs/tables/topk_ablation_results.csv` | Top-k subject head ablation results |
| `outputs/tables/random_ablation_results.csv` | Random head ablation control |
| `outputs/tables/low_ablation_results.csv` | Low-attention head ablation control |
| `outputs/tables/ablation_comparison.csv` | All ablation types merged |
| `outputs/tables/scatter_attention_vs_probdrop.csv` | Per-(sample, head) attention score and probability drop |
| `outputs/tables/head_activation_patching.csv` | Head patching recovery scores |
| `outputs/tables/causal_tracing_residual_patching.csv` | Residual stream patching recovery scores |
| `outputs/tables/top_causal_heads.csv` | Top-5 causal heads per target |
| `outputs/tables/top_causal_tracing_positions.csv` | Top-10 (layer, position) pairs per target |
| `results/filtering_summary.csv` | Ablation stats split by GPT-2 correctness |
| `results/relation_type_summary.csv` | Ablation stats per relation type |
| `results/control_comparison_summary.csv` | Subject vs. random vs. low-attention summary |
| `results/statistical_tests.csv` | Statistical test results |

---

## References

1. Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT*. NeurIPS 2022. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
2. Elhage, N., et al. (2021). *A Mathematical Framework for Transformer Circuits*. Anthropic. [link](https://transformer-circuits.pub/2021/framework/index.html)
3. Nanda, N., & Bloom, J. (2022). *TransformerLens*. [GitHub](https://github.com/neelnanda-io/TransformerLens)
4. Meng, K., et al. (2023). *Mass-Editing Memory in a Transformer*. ICLR 2023. [arXiv:2210.07229](https://arxiv.org/abs/2210.07229)

---

## License

This project is for academic purposes (CENG467 — Term Project). All datasets used are publicly available under their respective licenses.
