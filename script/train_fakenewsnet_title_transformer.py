from pathlib import Path
import sys

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.project_config import ARTIFACTS_DIR, DATA_DIR, MODEL_PATHS, MODELS_DIR
from utils.transformer_training import TextClassificationDataset


LABELS = ["FAKE", "REAL"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_LENGTH = 128
EPOCHS = 2.0
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
LIMIT = None
SAVE_FINAL_MODEL = False

MODEL_OUTPUT_DIR = MODELS_DIR / "fakenewsnet_title_model" / "distilbert_model"
TOKENIZER_OUTPUT_DIR = MODELS_DIR / "fakenewsnet_title_model" / "distilbert_tokenizer"
TRAINING_OUTPUT_DIR = ARTIFACTS_DIR / "training" / "fakenewsnet_title_transformer"


def load_fakenewsnet_titles(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    base_dir = data_dir / "FakeNewsNet"
    frames = []
    for file_name, label in [
        ("gossipcop_fake.csv", "FAKE"),
        ("politifact_fake.csv", "FAKE"),
        ("gossipcop_real.csv", "REAL"),
        ("politifact_real.csv", "REAL"),
    ]:
        frame = pd.read_csv(base_dir / file_name)
        frame = frame.assign(label=label, source_file=file_name)
        frames.append(frame)

    full_frame = pd.concat(frames, ignore_index=True)
    full_frame["title_text"] = full_frame["title"].fillna("").astype(str).str.strip()
    full_frame = full_frame.loc[full_frame["title_text"].str.len() > 0].copy()
    if LIMIT is not None:
        full_frame = full_frame.sample(
            n=min(LIMIT, len(full_frame)),
            random_state=RANDOM_STATE,
        )
    return full_frame.reset_index(drop=True)


def compute_metrics(eval_pred) -> dict:
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


def main() -> None:
    frame = load_fakenewsnet_titles()
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    labels = frame["label"].map(label_to_id).astype(int)

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        frame["title_text"].astype(str).tolist(),
        labels.tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    source_model_dir = MODEL_PATHS["fake_news"]["model"]
    source_tokenizer_dir = MODEL_PATHS["fake_news"]["tokenizer"]
    tokenizer = DistilBertTokenizerFast.from_pretrained(source_tokenizer_dir)
    model = DistilBertForSequenceClassification.from_pretrained(source_model_dir)
    model.config.id2label = {index: label for index, label in enumerate(LABELS)}
    model.config.label2id = {label: index for index, label in enumerate(LABELS)}

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    eval_encodings = tokenizer(
        eval_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    eval_dataset = TextClassificationDataset(eval_encodings, eval_labels)

    TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(TRAINING_OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)

    if SAVE_FINAL_MODEL:
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TOKENIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(TOKENIZER_OUTPUT_DIR)

    print(f"Saved FakeNewsNet title transformer training outputs to: {TRAINING_OUTPUT_DIR}")
    print(metrics)


if __name__ == "__main__":
    main()
