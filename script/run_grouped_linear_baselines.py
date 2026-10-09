from utils.classical_baselines import baseline_models, compute_classification_metrics
from utils.datasets import (
    GROUPED_SPLIT_VERSION,
    load_fake_news_kaggle_bundle,
    load_stance_detection_bundle,
)
from utils.transformer_training import prepare_training_split


TEST_SIZE = 0.2
RANDOM_STATE = 42


def run_task(task_name):
    if task_name == "fake_news":
        data_path = "data/preprocessed/fakenews_preprocessed.csv"
        text_column = "prep_text"
        label_column = "real"
        bundle = load_fake_news_kaggle_bundle()
        label_names = ["FAKE", "REAL"]
        label_lookup = {0: "FAKE", 1: "REAL"}
    elif task_name == "stance":
        data_path = "data/preprocessed/stance_preprocessed.csv"
        text_column = "combined_text"
        label_column = "stance_label"
        bundle = load_stance_detection_bundle()
        label_names = bundle.labels
        label_lookup = None
    else:
        raise ValueError(f"Unknown task: {task_name}")

    train_frame, _ = prepare_training_split(
        task_name=task_name,
        data_path=data_path,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_texts = train_frame[text_column].astype(str)
    train_labels = train_frame[label_column]
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
        }
    )
    print(
        f"{task_name}: accuracy {metrics['accuracy']:.4f}, "
        f"macro F1 {metrics['macro_f1']:.4f}"
    )
    return metrics


def main():
    run_task("fake_news")
    run_task("stance")


if __name__ == "__main__":
    main()
