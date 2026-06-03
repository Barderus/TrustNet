from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.project_config import DATA_DIR


FAKE_NEWS_LABELS = ["FAKE", "REAL"]
STANCE_LABELS = ["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    label_column: str
    text_column: str
    labels: list[str]


def _combine_title_and_text(frame: pd.DataFrame) -> pd.Series:
    title = frame["title"].fillna("").astype(str).str.strip()
    text = frame["text"].fillna("").astype(str).str.strip()
    return (title + "\n\n" + text).str.strip()


def load_fake_news_kaggle_bundle(
    data_dir: Path = DATA_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    fake_path = data_dir / "fake-news-kaggle" / "Fake.csv"
    true_path = data_dir / "fake-news-kaggle" / "True.csv"

    fake_frame = pd.read_csv(fake_path)
    true_frame = pd.read_csv(true_path)

    fake_frame = fake_frame.assign(label="FAKE")
    true_frame = true_frame.assign(label="REAL")

    full_frame = pd.concat([fake_frame, true_frame], ignore_index=True)
    full_frame["input_text"] = _combine_title_and_text(full_frame)
    full_frame = full_frame.loc[full_frame["input_text"].str.len() > 0].copy()

    train_frame, test_frame = train_test_split(
        full_frame,
        test_size=test_size,
        random_state=random_state,
        stratify=full_frame["label"],
    )

    train_frame = train_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    return DatasetBundle(
        name="fake-news-kaggle",
        train=train_frame,
        test=test_frame,
        label_column="label",
        text_column="input_text",
        labels=FAKE_NEWS_LABELS,
    )


def _load_stance_pair_frame(bodies_path: Path, stances_path: Path) -> pd.DataFrame:
    bodies = pd.read_csv(bodies_path)
    stances = pd.read_csv(stances_path)

    merged = stances.merge(bodies, on="Body ID", how="inner")
    merged["label"] = merged["Stance"].str.upper()
    merged["headline"] = merged["Headline"].fillna("").astype(str).str.strip()
    merged["body"] = merged["articleBody"].fillna("").astype(str).str.strip()
    merged["input_text"] = (merged["headline"] + " [SEP] " + merged["body"]).str.strip()
    return merged


def load_stance_detection_bundle(data_dir: Path = DATA_DIR) -> DatasetBundle:
    base_dir = data_dir / "StanceDetection"
    train_frame = _load_stance_pair_frame(
        base_dir / "train_bodies.csv",
        base_dir / "train_stances.csv",
    ).reset_index(drop=True)
    test_frame = _load_stance_pair_frame(
        base_dir / "competition_test_bodies.csv",
        base_dir / "competition_test_stances.csv",
    ).reset_index(drop=True)

    return DatasetBundle(
        name="stance-detection",
        train=train_frame,
        test=test_frame,
        label_column="label",
        text_column="input_text",
        labels=STANCE_LABELS,
    )
