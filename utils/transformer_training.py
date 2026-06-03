from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from utils.project_config import DATA_DIR, MODEL_PATHS


@dataclass(frozen=True)
class TransformerTrainingConfig:
    task_name: str
    data_path: Path
    text_column: str
    label_column: str
    label_names: list[str]
    model_output_dir: Path
    tokenizer_output_dir: Path
    training_output_dir: Path
    test_size: float = 0.2
    random_state: int = 42
    pretrained_model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    num_train_epochs: float = 3.0
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    limit: int | None = None


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index: int) -> dict:
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def fake_news_training_config(output_base_dir: Path) -> TransformerTrainingConfig:
    paths = MODEL_PATHS["fake_news"]
    return TransformerTrainingConfig(
        task_name="fake_news",
        data_path=DATA_DIR / "preprocessed" / "fakenews_preprocessed.csv",
        text_column="prep_text",
        label_column="real",
        label_names=["FAKE", "REAL"],
        model_output_dir=paths["model"],
        tokenizer_output_dir=paths["tokenizer"],
        training_output_dir=output_base_dir / "training" / "fake_news_transformer",
    )


def stance_training_config(output_base_dir: Path) -> TransformerTrainingConfig:
    paths = MODEL_PATHS["stance"]
    return TransformerTrainingConfig(
        task_name="stance",
        data_path=DATA_DIR / "preprocessed" / "stance_preprocessed.csv",
        text_column="combined_text",
        label_column="stance_label",
        label_names=["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"],
        model_output_dir=paths["model"],
        tokenizer_output_dir=paths["tokenizer"],
        training_output_dir=output_base_dir / "training" / "stance_transformer",
    )


def _load_training_frame(config: TransformerTrainingConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.data_path)
    if config.text_column not in frame.columns:
        if config.task_name == "stance" and {"headline_prep", "body_prep"}.issubset(frame.columns):
            frame[config.text_column] = (
                frame["headline_prep"].fillna("").astype(str)
                + " [SEP] "
                + frame["body_prep"].fillna("").astype(str)
            )
        else:
            raise ValueError(
                f"Missing text column '{config.text_column}' in {config.data_path}."
            )

    if config.label_column not in frame.columns:
        if config.task_name == "stance" and "Stance" in frame.columns:
            frame[config.label_column] = frame["Stance"].astype(str).str.upper()
        else:
            raise ValueError(
                f"Missing label column '{config.label_column}' in {config.data_path}."
            )

    frame = frame.dropna(subset=[config.text_column, config.label_column]).copy()
    if config.limit is not None:
        frame = frame.head(config.limit).copy()
    return frame


def _prepare_labels(frame: pd.DataFrame, config: TransformerTrainingConfig) -> pd.Series:
    label_to_id = {
        label_name: label_index
        for label_index, label_name in enumerate(config.label_names)
    }

    if config.task_name == "fake_news":
        return frame[config.label_column].astype(int)

    labels = frame[config.label_column].astype(str).str.upper().map(label_to_id)
    if labels.isna().any():
        unknown = sorted(frame.loc[labels.isna(), config.label_column].astype(str).unique())
        raise ValueError(f"Unknown labels for {config.task_name}: {unknown}")
    return labels.astype(int)


def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def train_transformer(config: TransformerTrainingConfig) -> Path:
    frame = _load_training_frame(config)
    labels = _prepare_labels(frame, config)

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        frame[config.text_column].astype(str).tolist(),
        labels.tolist(),
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(config.pretrained_model_name)
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=config.max_length,
    )
    eval_encodings = tokenizer(
        eval_texts,
        truncation=True,
        padding=True,
        max_length=config.max_length,
    )

    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    eval_dataset = TextClassificationDataset(eval_encodings, eval_labels)

    id_to_label = {
        label_index: label_name
        for label_index, label_name in enumerate(config.label_names)
    }
    label_to_id = {
        label_name: label_index
        for label_index, label_name in id_to_label.items()
    }
    model = DistilBertForSequenceClassification.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(config.label_names),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    config.training_output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(config.training_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)

    config.model_output_dir.mkdir(parents=True, exist_ok=True)
    config.tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.model_output_dir)
    tokenizer.save_pretrained(config.tokenizer_output_dir)
    return config.model_output_dir
