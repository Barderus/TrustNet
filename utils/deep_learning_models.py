from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class DeepLearningConfig:
    task_name: str
    data_path: Path
    text_column: str
    label_column: str
    label_names: list
    output_dir: Path
    max_vocab_size: int = 20000
    max_sequence_length: int = 200
    embedding_dim: int = 64
    hidden_dim: int = 64
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    test_size: float = 0.2
    random_state: int = 42
    limit: int | None = None


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocabulary, max_sequence_length):
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        token_ids = encode_text(
            self.texts[index],
            self.vocabulary,
            self.max_sequence_length,
        )
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_classes,
        kernel_sizes=(3, 4, 5),
        num_filters=100,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)
        pooled_outputs = []

        for convolution in self.convolutions:
            activated = torch.relu(convolution(embedded))
            pooled = torch.max(activated, dim=2).values
            pooled_outputs.append(pooled)

        features = torch.cat(pooled_outputs, dim=1)
        features = self.dropout(features)
        return self.classifier(features)


class BiLSTMTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        final_hidden = self.dropout(final_hidden)
        return self.classifier(final_hidden)


def tokenize(text):
    return re.findall(r"[a-zA-Z]+", str(text).lower())


def build_vocabulary(texts, max_vocab_size):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    most_common = counter.most_common(max_vocab_size - 2)
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in most_common:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text, vocabulary, max_sequence_length):
    token_ids = [
        vocabulary.get(token, vocabulary[UNK_TOKEN])
        for token in tokenize(text)
    ][:max_sequence_length]

    padding_needed = max_sequence_length - len(token_ids)
    if padding_needed > 0:
        token_ids.extend([vocabulary[PAD_TOKEN]] * padding_needed)
    return token_ids


def load_frame(config):
    frame = pd.read_csv(config.data_path)
    frame = frame.dropna(subset=[config.text_column, config.label_column]).copy()
    if config.limit is not None:
        frame = frame.head(config.limit).copy()
    return frame


def prepare_labels(frame, config):
    if config.task_name == "fake_news":
        return frame[config.label_column].astype(int).tolist()

    label_to_id = {
        label_name: label_index
        for label_index, label_name in enumerate(config.label_names)
    }
    labels = frame[config.label_column].astype(str).str.upper().map(label_to_id)
    if labels.isna().any():
        unknown = sorted(frame.loc[labels.isna(), config.label_column].astype(str).unique())
        raise ValueError(f"Unknown labels for {config.task_name}: {unknown}")
    return labels.astype(int).tolist()


def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()
    losses = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids)
            batch_probabilities = torch.softmax(logits, dim=1)
            batch_predictions = torch.argmax(logits, dim=1)
            probabilities.extend(batch_probabilities.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    return true_labels, predictions, probabilities


def calculate_roc_auc(true_labels, probabilities, label_names):
    try:
        if len(label_names) == 2:
            positive_class_probabilities = [
                probability_vector[1] for probability_vector in probabilities
            ]
            return float(roc_auc_score(true_labels, positive_class_probabilities))

        return float(
            roc_auc_score(
                true_labels,
                probabilities,
                labels=list(range(len(label_names))),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return None


def compute_metrics(true_labels, predictions, probabilities, label_names):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "roc_auc": calculate_roc_auc(true_labels, probabilities, label_names),
    }


def run_deep_learning_experiments(config):
    frame = load_frame(config)
    texts = frame[config.text_column].astype(str).tolist()
    labels = prepare_labels(frame, config)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    vocabulary = build_vocabulary(train_texts, config.max_vocab_size)
    train_dataset = TextDataset(
        train_texts,
        train_labels,
        vocabulary,
        config.max_sequence_length,
    )
    test_dataset = TextDataset(
        test_texts,
        test_labels,
        vocabulary,
        config.max_sequence_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_specs = ["text_cnn", "bidirectional_lstm"]

    results = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_specs:
        if model_name == "text_cnn":
            model = TextCNNClassifier(
                vocab_size=len(vocabulary),
                embedding_dim=config.embedding_dim,
                num_classes=len(config.label_names),
            ).to(device)
        else:
            model = BiLSTMTextClassifier(
                vocab_size=len(vocabulary),
                embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
                num_classes=len(config.label_names),
            ).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        epoch_losses = []
        for _ in range(config.epochs):
            epoch_loss = train_one_epoch(
                model,
                train_loader,
                loss_function,
                optimizer,
                device,
            )
            epoch_losses.append(epoch_loss)

        true_labels, predictions, probabilities = predict(model, test_loader, device)
        metrics = compute_metrics(
            true_labels,
            predictions,
            probabilities,
            config.label_names,
        )
        result = {
            "task": config.task_name,
            "model": model_name,
            "epochs": config.epochs,
            "device": str(device),
            "final_train_loss": epoch_losses[-1],
            **metrics,
        }
        results.append(result)

        model_output = config.output_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), model_output)

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(config.output_dir / "metrics.csv", index=False)
    (config.output_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2),
        encoding="utf-8",
    )
    return results_frame
