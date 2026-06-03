# Limitations and Future Work

TrustNet is useful as a learning and research project, but it should not be
presented as a finished misinformation detection system. The project can show
how different NLP models behave on specific datasets, but that is different from
verifying whether a real article is true or false.

## Dataset Limitations

Fake-news datasets often include strong signals that are not really about
truthfulness. A model may learn patterns tied to source, topic, time period,
writing style, formatting, or duplicated content.

The labels also simplify a complicated problem. A binary fake/real label does
not capture uncertainty, satire, partial truth, changing events, or the
credibility of a source over time.

## Generalization Concerns

Strong performance on one dataset does not automatically mean the model will
work well on another dataset. A random split can place similar articles,
publishers, or writing patterns in both training and testing data.

For that reason, cross-dataset evaluation is one of the most important next
steps. It will give a more honest view of whether the model has learned
generalizable patterns or mostly dataset-specific shortcuts.

## Interpretability Limitations

Token-level explanations can help show which words influenced a prediction, but
they do not prove that the model reasoned correctly. Explanations can still
reflect spurious correlations, tokenization behavior, or artifacts in the
training data.

I treat explanations as a way to inspect model behavior, not as evidence that a
claim is true or false.

## Ethical Concerns

TrustNet should not be used as an automated fact-checker. A prediction from the
model is only a screening signal. In a real setting, this kind of system would
need clear uncertainty messaging, careful user interface design, and human
review.

There is also a risk of harm if a system labels content incorrectly. False
positives can unfairly flag reliable content, while false negatives can allow
misleading content to pass without review.

## Future Work

- Run cross-dataset evaluation on FakeNewsNet and other external datasets.
- Add grouped or source-aware splits when metadata allows.
- Improve duplicate and near-duplicate detection.
- Compare transformer results against strong TF-IDF baselines.
- Add calibration analysis and uncertainty thresholds.
- Add structured error analysis for false positives and false negatives.
- Add explainability examples for both correct and incorrect predictions.
- Integrate explainability into the Streamlit app.
- Explore human-in-the-loop review workflows.
