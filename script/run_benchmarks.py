from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from utils.datasets import (
    DatasetBundle,
    load_fake_news_kaggle_bundle,
    load_stance_detection_bundle,
)
from utils.evaluation import (
    build_predictions_frame,
    compute_metrics,
    create_run_directory,
    save_evaluation_outputs,
)
from utils.model_loader import load_fake_news_model, load_stance_model
from utils.prediction import predict_text
from utils.project_config import ARTIFACTS_DIR


def _evaluate_bundle(
    bundle: DatasetBundle,
    model,
    tokenizer,
    task_name: str,
    output_base_dir: Path,
    limit: int | None = None,
) -> Path:
    evaluation_frame = bundle.test.copy()
    if limit is not None:
        evaluation_frame = evaluation_frame.head(limit).copy()

    true_labels = evaluation_frame[bundle.label_column].tolist()
    predicted_indices: list[int] = []
    predicted_labels: list[str] = []
    probabilities: list[list[float]] = []

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

    output_dir = create_run_directory(output_base_dir, task_name, bundle.name)
    save_evaluation_outputs(
        output_dir=output_dir,
        predictions=predictions,
        metrics=metrics,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        labels=bundle.labels,
    )
    return output_dir


def run_fake_news_benchmark(output_base_dir: Path, limit: int | None) -> Path:
    bundle = load_fake_news_kaggle_bundle()
    model, tokenizer = load_fake_news_model()
    return _evaluate_bundle(
        bundle=bundle,
        model=model,
        tokenizer=tokenizer,
        task_name="fake_news",
        output_base_dir=output_base_dir,
        limit=limit,
    )


def run_stance_benchmark(output_base_dir: Path, limit: int | None) -> Path:
    bundle = load_stance_detection_bundle()
    model, tokenizer = load_stance_model()
    return _evaluate_bundle(
        bundle=bundle,
        model=model,
        tokenizer=tokenizer,
        task_name="stance",
        output_base_dir=output_base_dir,
        limit=limit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrustNet evaluation benchmarks.")
    parser.add_argument(
        "--task",
        choices=["fake_news", "stance", "all"],
        default="all",
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ARTIFACTS_DIR),
        help="Base directory where evaluation artifacts will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of evaluation examples to run for quick checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_base_dir = Path(args.output_dir)

    if args.task in {"fake_news", "all"}:
        fake_news_dir = run_fake_news_benchmark(output_base_dir, args.limit)
        print(f"Fake news evaluation artifacts saved to: {fake_news_dir}")

    if args.task in {"stance", "all"}:
        stance_dir = run_stance_benchmark(output_base_dir, args.limit)
        print(f"Stance evaluation artifacts saved to: {stance_dir}")


if __name__ == "__main__":
    main()
