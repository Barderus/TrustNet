import json
from pathlib import Path
import sys

from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.classical_baselines import baseline_models, compute_classification_metrics
from utils.datasets import (
    GROUPED_SPLIT_VERSION,
    load_fake_news_kaggle_bundle,
    load_stance_detection_bundle,
)
from utils.evaluation import create_run_directory
from utils.project_config import ARTIFACTS_DIR
from utils.transformer_training import (
    fake_news_training_config,
    prepare_training_split,
    stance_training_config,
)


def run_task(task_name: str) -> None:
    if task_name == "fake_news":
        config = fake_news_training_config(ARTIFACTS_DIR)
        bundle = load_fake_news_kaggle_bundle()
        label_names = ["FAKE", "REAL"]
        label_lookup = {0: "FAKE", 1: "REAL"}
    elif task_name == "stance":
        config = stance_training_config(ARTIFACTS_DIR)
        bundle = load_stance_detection_bundle()
        label_names = bundle.labels
        label_lookup = None
    else:
        raise ValueError(f"Unknown task: {task_name}")

    train_frame, _ = prepare_training_split(config)
    train_texts = train_frame[config.text_column].astype(str)
    train_labels = train_frame[config.label_column]
    if label_lookup is not None:
        train_labels = train_labels.astype(int).map(label_lookup)
    else:
        train_labels = train_labels.astype(str).str.upper()

    test_texts = bundle.test[bundle.text_column].astype(str)
    test_labels = bundle.test[bundle.label_column].astype(str)
    model = baseline_models()["linear_svc_tfidf"]
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)
    scores = model.decision_function(test_texts)

    metrics = compute_classification_metrics(
        test_labels.tolist(), predictions.tolist(), label_names, scores
    )
    metrics.update(
        {
            "model": "linear_svc_tfidf",
            "split_protocol": GROUPED_SPLIT_VERSION,
            "train_examples": len(train_frame),
            "test_examples": len(bundle.test),
            "classification_report": classification_report(
                test_labels,
                predictions,
                labels=label_names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                test_labels, predictions, labels=label_names
            ).tolist(),
        }
    )
    output_dir = create_run_directory(
        ARTIFACTS_DIR, "grouped_baselines", bundle.name
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        f"{task_name}: accuracy {metrics['accuracy']:.4f}, "
        f"macro F1 {metrics['macro_f1']:.4f}; saved to {output_dir}"
    )


def main() -> None:
    run_task("fake_news")
    run_task("stance")


if __name__ == "__main__":
    main()
