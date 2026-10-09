import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from utils.datasets import (
    fake_news_group_keys,
    load_fake_news_kaggle_bundle,
    load_stance_detection_bundle,
    near_duplicate_eval_texts,
    stance_group_keys,
)


class TextClassificationDataset(torch.utils.data.Dataset):
    """Give the Transformers Trainer tokenized text and its label."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


def prepare_training_split(
    task_name, data_path, test_size=0.2, random_state=42, limit=None
):
    frame = pd.read_csv(data_path)

    if task_name == "fake_news":
        text_column = "prep_text"
        label_column = "real"
    elif task_name == "stance":
        text_column = "combined_text"
        label_column = "stance_label"
        has_clean_text = {"headline_prep", "body_prep"}.issubset(frame.columns)
        if text_column not in frame.columns and has_clean_text:
            frame[text_column] = (
                frame["headline_prep"].fillna("").astype(str)
                + " [SEP] "
                + frame["body_prep"].fillna("").astype(str)
            )
        if label_column not in frame.columns and "Stance" in frame.columns:
            frame[label_column] = frame["Stance"].astype(str).str.upper()
    else:
        raise ValueError(f"Unknown training task: {task_name}")

    for column in (text_column, label_column):
        if column not in frame.columns:
            raise ValueError(f"Missing column '{column}' in {data_path}.")

    frame = frame.dropna(subset=[text_column, label_column]).copy()
    if limit is not None:
        frame = frame.head(limit).copy()

    if task_name == "fake_news":
        kaggle = load_fake_news_kaggle_bundle(
            test_size=test_size,
            random_state=random_state,
        )
        frame["split_group"] = fake_news_group_keys(
            frame["clean_text"], frame[text_column]
        )
        frame = frame.loc[
            ~frame["split_group"].isin(set(kaggle.test["split_group"]))
        ].copy()
        test_texts = set(kaggle.test["model_input"])
        overlapping_texts = test_texts | near_duplicate_eval_texts(
            test_texts, frame[text_column].astype(str)
        )
        frame = frame.loc[~frame[text_column].isin(overlapping_texts)].copy()
        labels = frame[label_column].astype(int)
    else:
        competition = load_stance_detection_bundle()
        frame = frame.loc[
            frame["Body ID"].isin(set(competition.train["Body ID"]))
        ].copy()
        frame["split_group"] = stance_group_keys(
            frame["articleBody"], frame["Body ID"]
        )
        labels = frame[label_column].astype(str).str.upper()
        valid_labels = {"AGREE", "DISAGREE", "DISCUSS", "UNRELATED"}
        unknown = sorted(set(labels) - valid_labels)
        if unknown:
            raise ValueError(f"Unknown stance labels: {unknown}")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_indices, eval_indices = next(
        splitter.split(frame, labels, groups=frame["split_group"])
    )
    train_frame = frame.iloc[train_indices].copy()
    eval_frame = frame.iloc[eval_indices].copy()

    if task_name == "fake_news":
        train_texts = set(train_frame[text_column].astype(str))
        overlapping_texts = train_texts | near_duplicate_eval_texts(
            train_texts, eval_frame[text_column].astype(str)
        )
        eval_frame = eval_frame.loc[
            ~eval_frame[text_column].isin(overlapping_texts)
        ].copy()

    return train_frame, eval_frame
