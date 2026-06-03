# Data

This folder contains the source datasets used in TrustNet. I keep the raw data
organized here so the project has a clear starting point before preprocessing,
training, or evaluation.

Generated preprocessing outputs are not committed because they can be recreated
from the scripts.

## Dataset Folders

| Folder | Dataset | Used for |
| --- | --- | --- |
| `data/fake-news-kaggle/` | Kaggle Fake News Dataset | Main fake-news classification data |
| `data/StanceDetection/` | FNC-1 stance detection data | Headline/body stance detection |
| `data/FakeNewsNet/` | FakeNewsNet files | External comparison and generalization checks |
| `data/More-fake-news/` | Additional fake-news TSV data | Extra data used during exploration |
| `data/preprocessed/` | Generated preprocessing outputs | Recreated locally and ignored by Git |

## Generated Data

Preprocessed files are written under:

```text
data/preprocessed/
```

To recreate them, run this from the project root:

```bash
uv run python script/preprocess_data.py
```

## Notes

Each dataset comes from its original public source and may have its own license
or usage rules. Any reported result should name the dataset and split used to
produce it, especially when comparing fake-news detection and stance detection
results.
