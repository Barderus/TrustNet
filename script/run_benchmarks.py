from tqdm import tqdm

from utils.datasets import load_fake_news_kaggle_bundle, load_stance_detection_bundle
from utils.evaluation import (
    build_predictions_frame,
    compute_metrics,
    require_grouped_split_model,
)
from utils.model_loader import load_fake_news_model, load_stance_model
from utils.prediction import predict_text


TASK = "all"
LIMIT = None


def _evaluate_bundle(
    bundle,
    model,
    tokenizer,
    task_name,
    limit=None,
):
    evaluation_frame = bundle.test.copy()
    if limit is not None:
        evaluation_frame = evaluation_frame.head(limit).copy()

    true_labels = evaluation_frame[bundle.label_column].tolist()
    predicted_indices = []
    predicted_labels = []
    probabilities = []

    for input_text in tqdm(
        evaluation_frame[bundle.text_column].tolist(),
        desc=f"Evaluating {task_name}",
    ):
        predicted_index, probability_vector, _ = predict_text(model, tokenizer, input_text)
        predicted_indices.append(int(predicted_index))
        probability_list = [float(value) for value in probability_vector]
        probabilities.append(probability_list)
        predicted_labels.append(bundle.labels[predicted_index])

    predictions = build_predictions_frame(
        examples=evaluation_frame,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        predicted_indices=predicted_indices,
        probabilities=probabilities,
        labels=bundle.labels,
    )
    metrics = compute_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        labels=bundle.labels,
    )

    return predictions, metrics


def run_fake_news_benchmark(limit=None):
    bundle = load_fake_news_kaggle_bundle()
    model, tokenizer = load_fake_news_model()
    require_grouped_split_model(model)
    return _evaluate_bundle(
        bundle=bundle,
        model=model,
        tokenizer=tokenizer,
        task_name="fake_news",
        limit=limit,
    )


def run_stance_benchmark(limit=None):
    bundle = load_stance_detection_bundle()
    model, tokenizer = load_stance_model()
    require_grouped_split_model(model)
    return _evaluate_bundle(
        bundle=bundle,
        model=model,
        tokenizer=tokenizer,
        task_name="stance",
        limit=limit,
    )


def main():
    if TASK in {"fake_news", "all"}:
        _, metrics = run_fake_news_benchmark(LIMIT)
        print(
            f"Fake news: accuracy {metrics['accuracy']:.4f}, "
            f"macro F1 {metrics['macro_f1']:.4f}"
        )

    if TASK in {"stance", "all"}:
        _, metrics = run_stance_benchmark(LIMIT)
        print(
            f"Stance: accuracy {metrics['accuracy']:.4f}, "
            f"macro F1 {metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
