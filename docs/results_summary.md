# Results Summary

## Saved benchmark scores

The table below comes from saved runs of the grouped-split DistilBERT models. The revised training and evaluation notebooks have not been rerun, so these are the last recorded benchmark results.

| Evaluation | Examples | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Kaggle fake news, title and body | 8,983 | 0.9969 | 0.9969 |
| FNC-1 competition stance | 25,413 | 0.8912 | 0.6113 |
| Kaggle fake news, title only | 8,983 | 0.8316 | 0.8259 |
| FakeNewsNet titles, Kaggle-trained model | 23,196 | 0.2695 | 0.2344 |

## What the scores show

The Kaggle title-and-body result is very high, but it describes performance on that grouped Kaggle holdout. It does not show that the model can verify news or work equally well on other publishers. Source and writing-style cues may still help it classify articles.

Stance detection is harder than its accuracy suggests. The model correctly identified only 61 of 697 **DISAGREE** pairs. Most competition pairs are **UNRELATED**, so accuracy alone hides how often the model misses a less common stance. Per-class recall makes that weakness clear.

Removing article bodies also changes the fake-news result. On the same Kaggle holdout, title-only macro F1 fell to 0.8259. On FakeNewsNet titles, the Kaggle-trained model performed much worse and identified only 641 of 17,441 real-labeled titles. That check changes both the dataset and the input format, so the drop cannot be assigned to either change alone. Its predicted probabilities should not be read as certainty that a title is fake.

## Comparison and next reading

On the same grouped benchmark examples, a TF-IDF Linear SVC reached 0.9893 macro F1 for Kaggle fake news and 0.2457 for FNC-1 stance. DistilBERT did better on both, especially stance. Older classical and deep-learning scores used different splits, so they are useful background rather than a direct ranking against these benchmarks.

Notebook 5.3 now contains an in-domain FakeNewsNet title comparison, but it has not produced new reported results yet. Training on FakeNewsNet labels answers a different question from applying the Kaggle model without adaptation. Throughout the project, **FAKE**, **REAL**, and stance predictions describe dataset patterns; they are not fact checks.
