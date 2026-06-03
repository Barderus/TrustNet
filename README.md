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

## Main Questions

The project is guided by three questions:

1. How well do fake-news detection models perform on the datasets used here?
2. Can stance detection add useful context when reviewing misinformation?
3. What kinds of mistakes do the models make, and what can explanations show
   about those mistakes?

## What The Project Includes

| Component | Purpose |
| --- | --- |
| Fake-news detection | Predicts whether an article resembles fake or real examples from the training data |
| Stance detection | Predicts whether a headline and article body agree, disagree, discuss the same topic, or are unrelated |
| Streamlit app | Provides a simple interface for entering text and viewing predictions |
| Preprocessing scripts | Convert raw datasets into model-ready inputs |
| Training scripts | Train classical, deep learning, and transformer-based models |
| Evaluation scripts | Save metrics, predictions, confusion matrices, and calibration plots |

## Datasets

| Dataset | Use |
| --- | --- |
| Kaggle Fake News Dataset | Main fake-news classification dataset |
| FNC-1 | Main stance detection dataset |
| FakeNewsNet | Planned external comparison dataset |
| More Fake News | Additional fake-news data used during exploration |

More detail about the local data layout is in [data/README.md](data/README.md).

## Project Structure

```text
TrustNet/
  data/        raw datasets and dataset notes
  docs/        methodology, results, and limitations
  notebooks/   exploration, model review, and analysis notebooks
  script/      preprocessing, training, and evaluation commands
  streamlit/   Streamlit user interface
  utils/       shared Python utilities
  models/      local model artifacts, not included by default
```

## Setup

Install the project dependencies with `uv`:

```bash
uv sync
```

If dependencies change, refresh the lockfile:

```bash
uv lock
```

## Common Commands

Create preprocessing outputs:

```bash
uv run python script/preprocess_data.py
```

Run classical TF-IDF baselines:

```bash
uv run python script/run_classical_baselines.py
```

Train TextCNN and Bidirectional LSTM models:

```bash
uv run python script/train_fake_news_deep_learning.py
uv run python script/train_stance_deep_learning.py
```

Fine-tune transformer models:

```bash
uv run python script/train_fake_news_transformer.py
uv run python script/train_stance_transformer.py
```

Run benchmark evaluation after model artifacts exist:

```bash
uv run python script/run_benchmarks.py
```

Run the Streamlit app after model artifacts exist:

```bash
uv run streamlit run streamlit/app.py
```

## Model Artifacts

Large trained model files are not included by default. The app expects local
model artifacts under:

```text
models/
  fake_news_model/
    distilbert_fakenews_model/
    distilbert_fakenews_tokenizer/
  stance_detection_model/
    distilbert_stanceD/
    distilbert_tokenizer_stanceD/
```

The model paths are defined in `utils/project_config.py`.

## Generated Outputs

Some files are generated locally and are ignored by Git:

- `data/preprocessed/`
- `artifacts/`
- generated charts or HTML files under `images/`

These files can be recreated by running the scripts above.

## Documentation

The main project writeups are:

- [Methodology](docs/methodology.md)
- [Results Summary](docs/results_summary.md)
- [Limitations and Future Work](docs/limitations_and_future_work.md)

## Current Status

TrustNet is being cleaned up so the main workflow is easier to understand and
repeat. The notebooks are still useful for exploration and interpretation, but
the core preprocessing, training, and evaluation steps are moving into scripts.

The current priorities are:

- keep the project reproducible from a clean setup
- compare classical, deep learning, and transformer models clearly
- add cross-dataset evaluation
- document errors and limitations honestly
- add explainability examples that help interpret model behavior

## Important Limitations

TrustNet should not be treated as an automated fact-checker. The models learn
patterns from the datasets they are trained on, and those patterns may not hold
for new sources, topics, or time periods.

The output should be read as a model prediction based on training data patterns,
not as proof that an article is true or false.
