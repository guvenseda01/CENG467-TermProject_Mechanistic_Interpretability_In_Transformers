# Mechanistic Interpretability in Transformers
### CENG467 — Term Project | Group 7

> Analyzing attention head behavior and performing causal tracing experiments to understand how factual knowledge emerges in large language models.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Results](#results)
- [References](#references)

---

## Overview

This project investigates the internal mechanisms of Transformer-based language models (specifically GPT-2) through the lens of **mechanistic interpretability**. We focus on two core techniques:

| Technique | Goal |
|---|---|
| **Attention Head Analysis** | Identify what each attention head attends to and classify head roles (e.g., induction heads, name-mover heads, duplicate-token heads) |
| **Causal Tracing** | Localize where factual knowledge is stored by patching activations and measuring the effect on model predictions |

The experiments are conducted on the **CounterFact** and **SQuAD** datasets, which provide factual question-answer pairs and counterfactual edit pairs respectively.

---

## Project Structure

```
CENG467-TermProject_Mechanistic_Interpretability_In_Transformers/
│
├── CENG467_Group7_Final_Version.ipynb   # Main experiment notebook
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
└── src/
    ├── attention_analysis.py            # Attention head analysis utilities
    ├── causal_tracing.py                # Causal tracing / activation patching
    ├── dataset_utils.py                 # Dataset loading and preprocessing
    └── visualization.py                 # Plotting and visualization helpers
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended (experiments run on CPU but will be slow)

### Installation

```bash
# Clone the repository
git clone https://github.com/guvenseda01/CENG467-TermProject_Mechanistic_Interpretability_In_Transformers.git
cd CENG467-TermProject_Mechanistic_Interpretability_In_Transformers

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook CENG467_Group7_Final_Version.ipynb
```

Or open it directly in Google Colab (recommended for GPU access):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guvenseda01/CENG467-TermProject_Mechanistic_Interpretability_In_Transformers/blob/main/CENG467_Group7_Final_Version.ipynb)

---

## Datasets

### CounterFact
- A benchmark dataset of factual statements paired with counterfactual alternatives (e.g., *"The Eiffel Tower is in Paris"* → *"The Eiffel Tower is in Rome"*).
- Used for **causal tracing** to localize where factual associations are stored in the model.
- Source: [rome-beats/counterfact](https://huggingface.co/datasets/NeelNanda/counterfact-tracing)

### SQuAD
- Stanford Question Answering Dataset — reading comprehension questions over Wikipedia passages.
- Used for **attention analysis** to observe how heads route information for factual recall.
- Source: [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad)

---

## Experiments

### 1. Attention Head Analysis

We extract and visualize the attention patterns of all heads in GPT-2 across different token positions.

Key steps:
- Hook into each attention layer using TransformerLens
- Compute **attention rollout** and direct **QK attention scores**
- Classify heads by their behavior patterns:
  - **Induction Heads**: Copy previous token patterns (Layer 5–6)
  - **Name-Mover Heads**: Move subject tokens to the output position
  - **Duplicate Token Heads**: Detect repeated tokens in context

```python
from src.attention_analysis import load_model, get_attention_patterns, classify_heads

model = load_model("gpt2")
patterns = get_attention_patterns(model, prompt="The Eiffel Tower is located in")
head_labels = classify_heads(patterns)
```

### 2. Causal Tracing

We implement the **activation patching** technique from the ROME paper (Meng et al., 2022) to trace where factual knowledge is encoded.

Key steps:
- Run the model on a **clean** prompt (factual) and a **corrupted** prompt (noisy subject tokens)
- Patch activations layer by layer from the clean run into the corrupted run
- Measure the **indirect effect** on the target token probability

```python
from src.causal_tracing import run_causal_trace

results = run_causal_trace(
    model=model,
    tokenizer=tokenizer,
    prompt="The capital of France is",
    subject="France",
    target=" Paris"
)
# results contains indirect effect scores per (layer, position)
```

The heatmaps produced show that factual knowledge is most strongly localized in **early-to-mid MLP layers** (layers 3–8 in GPT-2).

---

## Results

| Experiment | Key Finding |
|---|---|
| Attention Head Classification | Induction heads reliably detected at layers 5–6; name-mover heads at layers 9–11 |
| Causal Tracing (CounterFact) | Highest indirect effect at MLP layers 3–8, subject's last token position |
| SQuAD Attention Rollout | Later layers show strong subject-to-answer token attention for factual queries |

Visualizations (attention heatmaps and causal trace plots) are generated inline in the notebook.

---

## References

1. Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT*. NeurIPS 2022. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
2. Elhage, N., et al. (2021). *A Mathematical Framework for Transformer Circuits*. Anthropic. [link](https://transformer-circuits.pub/2021/framework/index.html)
3. Nanda, N., & Bloom, J. (2022). *TransformerLens*. [GitHub](https://github.com/neelnanda-io/TransformerLens)
4. Rajpurkar, P., et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. EMNLP 2016.
5. Meng, K., et al. (2023). *Mass-Editing Memory in a Transformer*. ICLR 2023. [arXiv:2210.07229](https://arxiv.org/abs/2210.07229)

---

## License

This project is for academic purposes (CENG467 — Term Project). All datasets used are publicly available under their respective licenses.
