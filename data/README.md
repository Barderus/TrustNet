# Data

The raw dataset files below are tracked in this repository. Their source links
are provided for provenance or replacement if a local file is missing.

## Source Files

| Source | Local files | Use |
| --- | --- | --- |
| [Kaggle Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) | `data/fake-news-kaggle/Fake.csv`, `data/fake-news-kaggle/True.csv` | Main fake-news preprocessing |
| [FNC-1](https://github.com/FakeNewsChallenge/fnc-1) | `data/StanceDetection/train_bodies.csv`, `data/StanceDetection/train_stances.csv` | Stance preprocessing |
| [FNC-1](https://github.com/FakeNewsChallenge/fnc-1) | `data/StanceDetection/competition_test_bodies.csv`, `data/StanceDetection/competition_test_stances.csv` | Stance benchmark evaluation |
| [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) | `data/FakeNewsNet/gossipcop_fake.csv`, `data/FakeNewsNet/gossipcop_real.csv`, `data/FakeNewsNet/politifact_fake.csv`, `data/FakeNewsNet/politifact_real.csv` | External and in-domain title evaluation |
| [More Fake News](https://huggingface.co/datasets/Pulk17/Fake-News-Detection-dataset) | `data/More-fake-news/train.tsv` | Included in current fake-news preprocessing; required by the fake-news EDA notebook |

## Notes
Each dataset may have its own license or usage rules. Reported results should
name the dataset and split used to produce them.
