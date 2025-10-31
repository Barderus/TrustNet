# TrustNet: FakeNews Finder
## Overview

TrustNet (FakeNews Finder) is an interactive web tool that helps users check whether a news article or text appears credible.
Users can paste an article or link, and the system produces a trustworthiness score (0–100) along with an explanation of why the content was judged that way.
The goal is not only to detect misinformation but to teach critical reading by showing which words, tones, and claims influenced the decision.

This project was developed as part of my undergraduate Capstone in Computer Science.
It combines fake-news classification, stance detection, and explainability to make AI-based fact-checking more transparent.

## Motivation
As reported by Katie Langin in “Fake News Spreads Faster than True News on Twitter”, misinformation spreads farther and faster than verified facts, threatening democracy, free debate, and public trust.
While many fake-news models achieve high accuracy, they often fail to explain why.
TrustNet addresses this gap by providing interpretable results with uncertainty ranges and visual explanations that encourage users to reason critically.


## Proposed Model Architecture

| **Component** | **Description** |
|----------------|----------------|
| **Fake News Classifier** | Baselines: Logistic Regression & Linear SVM on TF-IDF.<br>Transformers: DistilBERT and RoBERTa fine-tuned for binary / 3-class classification. |
| **Stance Detection Model** | Sentence-pair encoding using DistilBERT ([CLS] token pooling) to predict stance. |
| **Explainability Layer** | SHAP values for linear models + token attribution (Gradient × Input / Attention Rollout) for transformers. |
| **Web Interface** | Streamlit or React frontend + FastAPI backend for asynchronous inference and visualization. |

##  Datasets

| **Source** | **Description** | **Link** |
|-------------|-----------------|-----------|
| **Kaggle – Fake News Dataset** | News articles labeled as real / fake. | [🔗 Dataset](https://www.kaggle.com/datasets/hassanamin/textdb3) |
| **Hugging Face – Fake News English** | Community dataset of fake / real news texts. | [🔗 Dataset](https://huggingface.co/datasets/community-datasets/fake_news_english) |
| **GitHub – FNC-1 (Fake News Challenge)** | Headline–body pairs with stance labels. | [🔗 Dataset](https://github.com/FakeNewsChallenge/fnc-1) |
| **GitHub – FakeNewsNet** | Articles + social context from PolitiFact and GossipCop. | [🔗 Dataset](https://github.com/KaiDMML/FakeNewsNet) |
