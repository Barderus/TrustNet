# TrustNet: FakeNews Finder

## Overview

TrustNet is an interactive NLP project focused on fake-news detection and stance detection.
Users can submit text and receive a model prediction with class probabilities through a Streamlit interface.

This project was developed as part of an undergraduate Computer Science capstone and is being upgraded into a more reproducible, interview-ready ML system.

## Motivation

Misinformation spreads quickly and is difficult to assess at scale. TrustNet explores how NLP models can help classify misleading content while also surfacing enough context for users to reason about results critically.

## Current Architecture

| Component | Description |
| --- | --- |
| Fake-news classifier | DistilBERT-based binary classification for fake vs real news text |
| Stance detection model | DistilBERT-based stance prediction for headline/body pairs |
| Streamlit app | User-facing interface for entering text and viewing predictions |
| Utilities | Shared preprocessing, prediction, path resolution, and model loading |

## Datasets

| Source | Description | Link |
| --- | --- | --- |
| Kaggle Fake News Dataset | News articles labeled as real or fake | https://www.kaggle.com/datasets/hassanamin/textdb3 |
| FNC-1 | Headline/body pairs with stance labels | https://github.com/FakeNewsChallenge/fnc-1 |
| FakeNewsNet | News articles with related misinformation context | https://github.com/KaiDMML/FakeNewsNet |

## Current Setup Notes

- The Streamlit app resolves project paths dynamically instead of relying on a machine-specific absolute path.
- Local model weights are not included in this repository.
- By default, the app expects model artifacts under `Models/`.
- You can override the model directory with the `TRUSTNET_MODELS_DIR` environment variable.

See [docs/model_artifacts.md](docs/model_artifacts.md) for the expected model folder layout and [docs/phase1_architecture.md](docs/phase1_architecture.md) for the current reproducibility-oriented structure.
