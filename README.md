# TrustNet: Misinformation and Stance Detection

## Overview

TrustNet is an NLP project I built to study how machine learning models handle
misinformation-related text. The project combines two related tasks:

- fake-news detection, where the model predicts whether an article looks closer
  to examples labeled fake or real
- stance detection, where the model predicts how a headline or claim relates to
  a longer article body


This started as my undergraduate Computer Science capstone. I am now cleaning it
up into a more reproducible project that is easier to review, rerun, and discuss
from both a research and applied machine learning perspective.

## Why I Built This

Misinformation is difficult to evaluate at scale, and simple true/false labels
do not capture the full problem. A model can help surface patterns in text, but
it should not be treated as a replacement for fact-checking or human judgment.

The goal of TrustNet is to compare several NLP approaches, understand where they
perform well, and be honest about where they fail. I am especially interested in
whether stance detection can add useful context and whether model explanations
can make predictions easier to interpret.

## Questions and Approach

The project is guided by three questions:

1. How well do fake-news models transfer across datasets and input formats?
2. Can stance detection add useful context when reviewing misinformation?
3. What kinds of mistakes do the models make, and what can explanations show
   about those mistakes?

I compare TF-IDF linear classifiers, TextCNN and Bidirectional LSTM models, and
DistilBERT. The Streamlit app uses the two saved DistilBERT models for predictions
and token-level explanations. These predictions reflect patterns in labeled
data; they do not verify whether an article is true.

## Datasets

| Dataset | Use |
| --- | --- |
| Kaggle Fake and Real News | Main title-and-body fake-news dataset |
| FNC-1 | Headline-and-body stance dataset |
| FakeNewsNet | External title-only evaluation and in-domain comparison |
| More Fake News | Optional additional data for preprocessing and exploration |

The raw dataset files used here are tracked in the repository. See the
[data guide](data/README.md) for exact files, sources, and generated inputs.

## Evaluation Status

The retrained DistilBERT models reached 0.9969 macro F1 on a grouped Kaggle
holdout and 0.6113 macro F1 on the FNC-1 competition set. The fake-news model
fell to 0.2344 macro F1 on FakeNewsNet titles without adaptation. Earlier
benchmark runs had known text overlap; their scores are retained as historical
results. The updated split audit found no shared groups, exact text matches, or
matches under its restricted near-duplicate screen. See the
[results summary](docs/results_summary.md) for protocols, per-class scores, and
limitations.