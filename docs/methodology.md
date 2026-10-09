# Methodology

## What TrustNet models

TrustNet handles two related tasks. The fake-news model learns the labels **FAKE** and **REAL** from articles; it does not check whether a claim is true. The stance model looks at a headline and an article body together and predicts whether the body **AGREES**, **DISAGREES**, **DISCUSSES**, or is **UNRELATED** to the headline. A stance label describes that relationship, not the truth of either text.

## Preparing the inputs

The fake-news training notebooks combine article titles and bodies from the Kaggle dataset. Shared cleaning removes URLs and punctuation, lowercases the text, and filters English stop words. The stance notebooks clean the headline and body separately, then join them with `[SEP]`. Streamlit applies the same cleaning to text entered or uploaded by a user.

Separate saved DistilBERT models power the two app tasks. Their training inputs are capped at 512 tokens. For longer predictions, inference splits the text into overlapping tokenizer chunks and averages their class probabilities. Training and long-text inference therefore cover text differently.

## Training and evaluation

We keep benchmark examples apart from training. For fake news, a seed-42 grouped split puts articles with matching cleaned bodies together. Training also excludes holdout matches and examples caught by a limited near-duplicate screen. A separate grouped split supplies training validation; the Kaggle holdout supplies the benchmark score.

For stance detection, the FNC-1 competition pairs are the benchmark. Development examples with body text matching a competition body are removed, and the remaining development data is grouped by cleaned body for validation. This helps avoid showing the model the same article body on both sides of a split.

The notebooks train TF-IDF, TextCNN, bidirectional LSTM, and DistilBERT models and show classification reports and confusion matrices. The benchmark script prints summary scores without creating dated report folders. We focus on macro F1 alongside accuracy because the stance labels are unevenly represented.

## Checks beyond the main benchmarks

A Kaggle title-only check uses the same fake-news holdout with less text. A separate FakeNewsNet check applies the Kaggle-trained model to titles from another source without retraining. Notebook 5.3 explores training on FakeNewsNet titles, which is an in-domain comparison rather than the same transfer test. These checks help show where a strong score depends on the dataset or input format.
