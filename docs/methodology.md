# Methodology

This document explains how TrustNet is set up from a modeling and evaluation
perspective. The goal is to make the project easy to follow without requiring a
reader to open every notebook.

## Research Questions

TrustNet is organized around three questions:

1. How well do transformer-based models generalize across fake-news datasets?
2. Can stance detection add useful context when reviewing misinformation?
3. What errors do the models make, and how can explanations help make those
   predictions easier to interpret?

## Datasets

| Dataset | Task | Labels | Text fields | Current use |
| --- | --- | --- | --- | --- |
| Kaggle Fake News Dataset | Fake-news detection | Fake, Real | Title, text | Primary train/test source |
| FNC-1 Stance Detection | Stance detection | Agree, Disagree, Discuss, Unrelated | Headline, article body | Primary stance train/evaluation source |
| FakeNewsNet | Fake-news detection and external context | Fake, Real by source files | News title and metadata | Planned cross-dataset evaluation |
| More Fake News TSV | Fake-news detection and external context | Dataset-specific labels | TSV fields | Dataset inventory and robustness checks |

The repository-level data layout is described in `data/README.md`.

## Preprocessing

For fake-news detection, the project combines an article title and body into one
text field called `input_text`. Empty inputs are removed before splitting the
data.

For stance detection, the project merges stance labels with article bodies by
`Body ID`, normalizes labels to uppercase, and creates the model input in this
format:

```text
headline [SEP] body
```

This keeps the headline and article body together while still giving the model a
clear separator between the two parts.

## Models

### Classical Baselines

The classical models use TF-IDF features with linear classifiers. These models
are important because they give the project a practical baseline. If a
transformer model performs well but only slightly improves on a simpler model,
that is still useful information.

The high fake-news scores should be interpreted carefully. Random splits can
allow source, topic, formatting, or writing-style patterns to appear in both the
training and test sets.

### Deep Learning Models

The project also includes TextCNN and Bidirectional LSTM models. These sit
between the classical baselines and transformer models in complexity. They are
useful for comparing whether stronger sequence models improve performance before
moving to pretrained transformers.

### Transformer Models

The current transformer workflow uses DistilBERT sequence-classification models
for:

- fake-news detection
- stance detection

The Streamlit app and benchmark scripts load these models through
`utils/model_loader.py`.

## Evaluation Metrics

The evaluation utilities report:

- accuracy
- macro precision
- macro recall
- macro F1
- weighted precision
- weighted recall
- weighted F1
- confusion matrix
- log loss when probabilities are available
- one-vs-rest calibration curves

Macro F1 is especially important for stance detection because some stance labels
are much less common than others. Accuracy alone can hide weak performance on
minority classes.

## Stance Detection Context

Fake-news detection predicts whether an article resembles examples labeled fake
or real in the training data. Stance detection answers a different question: how
does a headline or claim relate to an article body?

That distinction matters. An article might discuss a claim without supporting
it, contradict it, or be unrelated to it. Stance detection can add context to a
review workflow, but it does not prove whether a claim is true.

## Reproducibility Notes

The current fake-news evaluation split uses:

- `test_size=0.2`
- `random_state=42`
- stratification by label

The stance evaluation uses the FNC-1 competition test files as the labeled
holdout when available.

Final reported results should come from saved benchmark artifacts under
`artifacts/evaluation/`, not from one-off notebook outputs.
