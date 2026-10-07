from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils.project_config import DATA_DIR
from utils.preprocessing import clean_for_model, prepare_stance_input


FAKE_NEWS_LABELS = ["FAKE", "REAL"]
STANCE_LABELS = ["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"]
GROUPED_SPLIT_VERSION = "grouped-v1"


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


def fake_news_group_keys(body_texts: pd.Series, model_inputs: pd.Series) -> pd.Series:
    groups = []
    for body, model_input in zip(body_texts, model_inputs):
        body_words = clean_for_model(body).split()
        if len(body_words) >= 40:
            key = "body-prefix:" + " ".join(body_words[:20])
        elif body_words:
            key = "body:" + " ".join(body_words)
        else:
            key = "input:" + clean_for_model(model_input)
        groups.append(sha256(key.encode("utf-8")).hexdigest())
    return pd.Series(groups, index=body_texts.index)


def stance_group_keys(body_texts: pd.Series, body_ids: pd.Series) -> pd.Series:
    groups = []
    for body, body_id in zip(body_texts, body_ids):
        cleaned_body = clean_for_model(body)
        key = "body:" + cleaned_body if cleaned_body else f"body-id:{body_id}"
        groups.append(sha256(key.encode("utf-8")).hexdigest())
    return pd.Series(groups, index=body_texts.index)


def near_duplicate_eval_texts(train_texts, eval_texts):
    """Screen long cleaned inputs with matching starts and similar token sets."""
    train_by_prefix = defaultdict(list)
    train_keys = set(train_texts)
    for text in train_keys:
        words = text.split()
        if len(words) >= 40:
            train_by_prefix[tuple(words[:20])].append((set(words), len(words)))

    near_texts = set()
    for text in set(eval_texts) - train_keys:
        words = text.split()
        if len(words) < 40:
            continue
        tokens = set(words)
        for other_tokens, other_length in train_by_prefix.get(tuple(words[:20]), ()):
            if abs(len(words) - other_length) > max(len(words), other_length) * 0.1:
                continue
            if len(tokens & other_tokens) / len(tokens | other_tokens) >= 0.9:
                near_texts.add(text)
                break
    return near_texts


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
    full_frame["model_input"] = full_frame["input_text"].apply(clean_for_model)
    full_frame["split_group"] = fake_news_group_keys(
        full_frame["text"], full_frame["model_input"]
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(
        splitter.split(full_frame, full_frame["label"], groups=full_frame["split_group"])
    )
    train_frame = full_frame.iloc[train_indices].reset_index(drop=True)
    test_frame = full_frame.iloc[test_indices].reset_index(drop=True)
    overlapping_test_texts = set(train_frame["model_input"]) | near_duplicate_eval_texts(
        train_frame["model_input"], test_frame["model_input"]
    )
    test_frame = test_frame.loc[
        ~test_frame["model_input"].isin(overlapping_test_texts)
    ].reset_index(drop=True)

    return DatasetBundle(
        name="fake-news-kaggle",
        train=train_frame,
        test=test_frame,
        label_column="label",
        text_column="model_input",
        labels=FAKE_NEWS_LABELS,
    )


def load_fake_news_kaggle_title_bundle(
    data_dir: Path = DATA_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    bundle = load_fake_news_kaggle_bundle(
        data_dir=data_dir,
        test_size=test_size,
        random_state=random_state,
    )

    train_frame = bundle.train.copy()
    test_frame = bundle.test.copy()
    train_frame["title_input"] = train_frame["title"].fillna("").astype(str).str.strip()
    test_frame["title_input"] = test_frame["title"].fillna("").astype(str).str.strip()
    train_frame = train_frame.loc[train_frame["title_input"].str.len() > 0].copy()
    test_frame = test_frame.loc[test_frame["title_input"].str.len() > 0].copy()
    train_frame["title_model_input"] = train_frame["title_input"].apply(clean_for_model)
    test_frame["title_model_input"] = test_frame["title_input"].apply(clean_for_model)

    return DatasetBundle(
        name="fake-news-kaggle-title-only",
        train=train_frame.reset_index(drop=True),
        test=test_frame.reset_index(drop=True),
        label_column=bundle.label_column,
        text_column="title_model_input",
        labels=bundle.labels,
    )


def _load_stance_pair_frame(bodies_path: Path, stances_path: Path) -> pd.DataFrame:
    bodies = pd.read_csv(bodies_path)
    stances = pd.read_csv(stances_path)

    merged = stances.merge(bodies, on="Body ID", how="inner")
    merged["label"] = merged["Stance"].str.upper()
    merged["headline"] = merged["Headline"].fillna("").astype(str).str.strip()
    merged["body"] = merged["articleBody"].fillna("").astype(str).str.strip()
    merged["input_text"] = (merged["headline"] + " [SEP] " + merged["body"]).str.strip()
    merged["model_input"] = merged.apply(
        lambda row: prepare_stance_input(row["headline"], row["body"]),
        axis=1,
    )
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
    train_frame["split_group"] = stance_group_keys(
        train_frame["body"], train_frame["Body ID"]
    )
    test_frame["split_group"] = stance_group_keys(
        test_frame["body"], test_frame["Body ID"]
    )
    train_frame = train_frame.loc[
        ~train_frame["split_group"].isin(set(test_frame["split_group"]))
    ].reset_index(drop=True)

    return DatasetBundle(
        name="stance-detection",
        train=train_frame,
        test=test_frame,
        label_column="label",
        text_column="model_input",
        labels=STANCE_LABELS,
    )
