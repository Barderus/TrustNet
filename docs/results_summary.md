# Results Summary

This page summarizes the benchmark results that are currently saved in
`artifacts/`. The numbers are useful, but they should be read with the project
limitations in mind, especially for fake-news detection where random splits can
make the task look easier than it may be on new sources.

Cross-dataset evaluation, error analysis, and explainability outputs are still
in progress.

## Model Comparison

| Task | Dataset | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Fake-news detection | Kaggle Fake News | Logistic Regression + TF-IDF | 0.9891 | 0.9897 | 0.9879 | 0.9888 | Classical baseline |
| Fake-news detection | Kaggle Fake News | Linear SVC + TF-IDF | 0.9935 | 0.9940 | 0.9927 | 0.9933 | Best classical baseline in saved run |
| Fake-news detection | Kaggle Fake News | Ridge Classifier + TF-IDF | 0.9926 | 0.9932 | 0.9917 | 0.9924 | Classical baseline |
| Fake-news detection | Kaggle Fake News | TextCNN | 0.9914 | 0.9909 | 0.9915 | 0.9912 | Deep learning baseline |
| Fake-news detection | Kaggle Fake News | Bidirectional LSTM | 0.9868 | 0.9855 | 0.9877 | 0.9865 | Deep learning baseline |
| Fake-news detection | Kaggle Fake News | DistilBERT | 0.9945 | 0.9946 | 0.9944 | 0.9945 | Latest saved benchmark evaluation |
| Stance detection | FNC-1 | Logistic Regression + TF-IDF | 0.8177 | 0.7051 | 0.4505 | 0.5096 | Classical baseline |
| Stance detection | FNC-1 | Linear SVC + TF-IDF | 0.8371 | 0.7081 | 0.5140 | 0.5752 | Best classical baseline in saved run |
| Stance detection | FNC-1 | Ridge Classifier + TF-IDF | 0.8258 | 0.6910 | 0.4837 | 0.5432 | Classical baseline |
| Stance detection | FNC-1 | TextCNN | 0.8315 | 0.6691 | 0.5228 | 0.5618 | Deep learning baseline |
| Stance detection | FNC-1 | Bidirectional LSTM | 0.8764 | 0.7033 | 0.6560 | 0.6774 | Deep learning baseline |
| Stance detection | FNC-1 | DistilBERT | 0.8308 | 0.5759 | 0.5501 | 0.5423 | Latest saved benchmark evaluation |
| Cross-dataset fake-news evaluation | FakeNewsNet or More Fake News | DistilBERT trained on Kaggle | TBD | TBD | TBD | TBD | Planned generalization test |

## Transformer Training Evaluation

The transformer training scripts also saved trainer evaluation outputs. I keep
these numbers separate from the benchmark table because they are useful for
checking the training run, but the benchmark table is the better place to report
final results.

| Task | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Fake-news detection | DistilBERT | 0.9981 | 0.9981 | 0.9981 | 0.9981 | `artifacts/training/fake_news_transformer/eval_results.json` |
| Stance detection | DistilBERT | 0.9737 | 0.8717 | 0.8650 | 0.8678 | `artifacts/training/stance_transformer/eval_results.json` |

## Interpretation

The fake-news scores are very high across classical, deep learning, and
transformer models. That is encouraging, but it also means the results need a
careful follow-up check. A random split may reward patterns that are specific to
the dataset, such as publisher style, article formatting, topic distribution, or
duplicate and near-duplicate text.

The stance results show a more difficult problem. Accuracy can look reasonable
while macro F1 stays much lower because minority stance labels are harder to
recover. For that reason, stance detection should be discussed with macro F1 and
per-class examples, not accuracy alone.

## Saved Artifacts

Benchmark runs save outputs under:

```text
artifacts/evaluation/
```

Expected files per run:

- `metrics.json`
- `predictions.csv`
- `confusion_matrix.png`
- `calibration.png`
- `summary.md`

Current saved benchmark runs:

- `artifacts/baselines/fake_news/20260602T205353Z/metrics.csv`
- `artifacts/baselines/stance/20260602T205507Z/metrics.csv`
- `artifacts/deep_learning/fake_news/metrics.csv`
- `artifacts/deep_learning/stance/metrics.csv`
- `artifacts/evaluation/fake_news/fake-news-kaggle/20260603T003238Z/`
- `artifacts/evaluation/stance/stance-detection/20260603T005603Z/`

## Next Result Tasks

- Add cross-dataset evaluation results.
- Add false-positive and false-negative examples.
- Add explainability examples for correct and incorrect predictions.
