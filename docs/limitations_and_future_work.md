# Limitations and Future Work

TrustNet is my undergraduate capstone study of model behavior on labeled
datasets. Its fake-news output is a prediction of dataset labels, and its
stance output describes a headline/body relationship. Neither verifies a
claim or establishes an article's truth.

## Dataset Limitations

Source, topic, time period, writing style, formatting, and repeated content can
all correlate with a dataset's labels. On the current Kaggle holdout, `reuters`
appears in 4,261 of 4,272 real inputs and 70 of 4,711 fake inputs. Removing
that token in a small paired diagnostic changed five row predictions, but this
does not measure dependence on all source cues. The grouped split and restricted
near-duplicate screen reduce known overlap without eliminating every related
article or shared source pattern.

Binary fake/real labels also omit uncertainty, satire, partial truth, changing
events, and source credibility over time.

## Generalization Concerns

The Kaggle-trained model reached 0.9969 macro F1 on its grouped title/body
holdout but 0.2344 on FakeNewsNet titles without adaptation. That evaluation
changes the source and removes article bodies at the same time. A Kaggle
title-only check reached 0.8259 macro F1, but it does not isolate the remaining
source and labeling differences. The FNC-1 stance model reached 0.6113 macro
F1 and recognized only 61 of 697 `DISAGREE` pairs.

## Interpretability Limitations

Token attributions show local influence on the selected prediction. They can
reflect spurious correlations or tokenization effects and cannot recover
missing article context. On FakeNewsNet, 16,530 of 16,944 transfer errors had
predicted-class probability at least 0.9. The saved probabilities should not
be treated as factual confidence or used as a portable decision threshold.

## Ethical Concerns

TrustNet should not be used as an automated fact-checker. Any review workflow
would need clear uncertainty messaging, appropriate context, and human review.

There is also a risk of harm if a system labels content incorrectly. False
positives can unfairly flag reliable content, while false negatives can allow
misleading content to pass without review.

## Future Work

- Evaluate external title/body datasets and source-aware splits where metadata permits.
- Improve near-duplicate screening and audit source/style cues beyond one token.
- Compare models on matched examples and text coverage across input lengths.
- Study calibration by source and input format before considering thresholds.
- Review more correct and incorrect cases, including long inputs and app attributions.
- Explore a human review workflow with clear limits on what each prediction means.
