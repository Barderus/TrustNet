from pathlib import Path
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.datasets import DatasetBundle, load_fake_news_kaggle_title_bundle
from utils.evaluation import (
    build_predictions_frame,
    compute_metrics,
    create_run_directory,
    save_evaluation_outputs,
)
from utils.model_loader import load_fake_news_model
from utils.project_config import ARTIFACTS_DIR


LIMIT = None
BATCH_SIZE = 64


def predict_batches(
    bundle: DatasetBundle,
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
) -> tuple[list[int], list[str], list[list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predicted_indices: list[int] = []
    predicted_labels: list[str] = []
    probabilities: list[list[float]] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Evaluating Kaggle titles"):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_probabilities = torch.softmax(outputs.logits, dim=1).cpu()

        for probability_vector in batch_probabilities:
            probability_list = [float(value) for value in probability_vector.tolist()]
            predicted_index = int(torch.tensor(probability_list).argmax().item())
            predicted_indices.append(predicted_index)
            predicted_labels.append(bundle.labels[predicted_index])
            probabilities.append(probability_list)

    return predicted_indices, predicted_labels, probabilities


def main() -> None:
    bundle = load_fake_news_kaggle_title_bundle()
    evaluation_frame = bundle.test.copy()
    if LIMIT is not None:
        evaluation_frame = evaluation_frame.sample(
            n=min(LIMIT, len(evaluation_frame)),
            random_state=42,
        ).reset_index(drop=True)

    model, tokenizer = load_fake_news_model()
    predicted_indices, predicted_labels, probabilities = predict_batches(
        bundle=bundle,
        model=model,
        tokenizer=tokenizer,
        texts=evaluation_frame[bundle.text_column].tolist(),
    )

    true_labels = evaluation_frame[bundle.label_column].tolist()
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

    output_dir = create_run_directory(ARTIFACTS_DIR, "title_only", bundle.name)
    save_evaluation_outputs(
        output_dir=output_dir,
        predictions=predictions,
        metrics=metrics,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        labels=bundle.labels,
    )

    print(f"Saved title-only evaluation artifacts to: {output_dir}")
    print(
        "Accuracy: "
        f"{metrics['accuracy']:.4f}, "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
