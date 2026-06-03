import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils.project_config import ARTIFACTS_DIR, DATA_DIR


@dataclass(frozen=True)
class BaselineDataset:
    task_name: str
    frame: pd.DataFrame
    text_column: str
    label_column: str
    labels: list


def load_fake_news_baseline_dataset(data_dir: Path = DATA_DIR) -> BaselineDataset:
    frame = pd.read_csv(data_dir / "preprocessed" / "fakenews_preprocessed.csv")
    frame = frame.dropna(subset=["prep_text", "real"]).copy()
    frame["real"] = frame["real"].astype(int)
    return BaselineDataset(
        task_name="fake_news",
        frame=frame,
        text_column="prep_text",
        label_column="real",
        labels=[0, 1],
    )


def load_stance_baseline_dataset(data_dir: Path = DATA_DIR) -> BaselineDataset:
    frame = pd.read_csv(data_dir / "preprocessed" / "stance_preprocessed.csv")
    frame = frame.dropna(subset=["headline_prep", "body_prep", "Stance"]).copy()
    frame["combined_text"] = (
        frame["headline_prep"].fillna("").astype(str)
        + " [SEP] "
        + frame["body_prep"].fillna("").astype(str)
    )
    frame["stance_label"] = frame["Stance"].astype(str).str.upper()
    return BaselineDataset(
        task_name="stance",
        frame=frame,
        text_column="combined_text",
        label_column="stance_label",
        labels=["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"],
    )


def baseline_models(random_state: int = 42) -> dict[str, Pipeline]:
    def vectorizer() -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
        )

    return {
        "logistic_regression_tfidf": Pipeline(
            [
                ("tfidf", vectorizer()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        "linear_svc_tfidf": Pipeline(
            [
                ("tfidf", vectorizer()),
                ("model", LinearSVC(random_state=random_state)),
            ]
        ),
        "ridge_classifier_tfidf": Pipeline(
            [
                ("tfidf", vectorizer()),
                ("model", RidgeClassifier()),
            ]
        ),
    }


def compute_classification_metrics(
    y_true: list,
    y_pred: list,
    labels: list,
    y_score=None,
) -> dict:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    metrics["roc_auc"] = calculate_roc_auc(y_true, y_score, labels)
    return metrics


def calculate_roc_auc(y_true: list, y_score, labels: list):
    if y_score is None:
        return None

    try:
        if len(labels) == 2:
            if len(y_score.shape) == 1:
                return float(roc_auc_score(y_true, y_score))
            return float(roc_auc_score(y_true, y_score[:, 1]))

        return float(
            roc_auc_score(
                y_true,
                y_score,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return None


def get_model_scores(model, x_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return None


def create_baseline_run_dir(
    task_name: str,
    output_base_dir: Path = ARTIFACTS_DIR,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_base_dir / "baselines" / task_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_baselines(
    dataset: BaselineDataset,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    limit: int | None = None,
) -> pd.DataFrame:
    frame = dataset.frame
    if limit is not None:
        frame = frame.head(limit).copy()

    x_train, x_test, y_train, y_test = train_test_split(
        frame[dataset.text_column],
        frame[dataset.label_column],
        test_size=test_size,
        random_state=random_state,
        stratify=frame[dataset.label_column],
    )

    results: list[dict] = []
    for model_name, model in baseline_models(random_state=random_state).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        scores = get_model_scores(model, x_test)
        metrics = compute_classification_metrics(
            y_true=y_test.tolist(),
            y_pred=predictions.tolist(),
            labels=dataset.labels,
            y_score=scores,
        )
        result = {
            "task": dataset.task_name,
            "model": model_name,
            "test_size": test_size,
            "random_state": random_state,
            **metrics,
        }
        results.append(result)

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metrics.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        (model_dir / "classification_report.txt").write_text(
            classification_report(
                y_test,
                predictions,
                labels=dataset.labels,
                zero_division=0,
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=dataset.labels),
            index=dataset.labels,
            columns=dataset.labels,
        ).to_csv(model_dir / "confusion_matrix.csv")

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(output_dir / "metrics.csv", index=False)
    return results_frame
