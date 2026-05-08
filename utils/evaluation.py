from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class EvaluationArtifacts:
    output_dir: Path
    metrics_json: Path
    predictions_csv: Path
    confusion_matrix_png: Path
    calibration_png: Path


def create_run_directory(base_dir: Path, task_name: str, dataset_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = base_dir / "evaluation" / task_name / dataset_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_predictions_frame(
    examples: pd.DataFrame,
    true_labels: list[str],
    predicted_labels: list[str],
    predicted_indices: list[int],
    probabilities: list[list[float]],
    labels: list[str],
) -> pd.DataFrame:
    predictions = examples.copy().reset_index(drop=True)
    predictions["true_label"] = true_labels
    predictions["predicted_label"] = predicted_labels
    predictions["predicted_index"] = predicted_indices

    for label_index, label_name in enumerate(labels):
        predictions[f"prob_{label_name.lower()}"] = [
            float(prob_vector[label_index]) for prob_vector in probabilities
        ]

    predictions["confidence"] = [
        float(max(prob_vector)) for prob_vector in probabilities
    ]
    predictions["is_correct"] = (
        predictions["true_label"] == predictions["predicted_label"]
    )
    return predictions


def compute_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
    probabilities: list[list[float]],
    labels: list[str],
) -> dict:
    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
    }

    try:
        metrics["log_loss"] = float(log_loss(true_labels, probabilities, labels=labels))
    except ValueError:
        metrics["log_loss"] = None

    return metrics


def save_confusion_matrix(
    true_labels: list[str],
    predicted_labels: list[str],
    labels: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_calibration_plot(
    true_labels: list[str],
    probabilities: list[list[float]],
    labels: list[str],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))

    for label_index, label_name in enumerate(labels):
        binary_truth = [1 if value == label_name else 0 for value in true_labels]
        label_probs = [float(prob_vector[label_index]) for prob_vector in probabilities]
        prob_true, prob_pred = calibration_curve(
            binary_truth,
            label_probs,
            n_bins=10,
            strategy="uniform",
        )
        if len(prob_true) == 0:
            continue
        plt.plot(prob_pred, prob_true, marker="o", label=label_name)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("One-vs-Rest Calibration Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)


def write_summary_markdown(metrics: dict, output_path: Path, labels: list[str]) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Weighted F1: {metrics['weighted_f1']:.4f}",
    ]
    if metrics.get("log_loss") is not None:
        lines.append(f"- Log loss: {metrics['log_loss']:.4f}")

    lines.extend(["", "## Labels", ""])
    for label in labels:
        lines.append(f"- {label}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_evaluation_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    metrics: dict,
    true_labels: list[str],
    predicted_labels: list[str],
    probabilities: list[list[float]],
    labels: list[str],
) -> EvaluationArtifacts:
    metrics_json = output_dir / "metrics.json"
    predictions_csv = output_dir / "predictions.csv"
    confusion_matrix_png = output_dir / "confusion_matrix.png"
    calibration_png = output_dir / "calibration.png"
    summary_md = output_dir / "summary.md"

    predictions.to_csv(predictions_csv, index=False)
    save_metrics_json(metrics, metrics_json)
    save_confusion_matrix(true_labels, predicted_labels, labels, confusion_matrix_png)
    save_calibration_plot(true_labels, probabilities, labels, calibration_png)
    write_summary_markdown(metrics, summary_md, labels)

    return EvaluationArtifacts(
        output_dir=output_dir,
        metrics_json=metrics_json,
        predictions_csv=predictions_csv,
        confusion_matrix_png=confusion_matrix_png,
        calibration_png=calibration_png,
    )
