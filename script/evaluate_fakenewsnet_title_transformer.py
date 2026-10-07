from pathlib import Path
import sys

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from script.train_fakenewsnet_title_transformer import (
    LABELS,
    RANDOM_STATE,
    TEST_SIZE,
    load_fakenewsnet_titles,
)
from utils.evaluation import (
    build_predictions_frame,
    compute_metrics,
    create_run_directory,
    save_evaluation_outputs,
)
from utils.project_config import ARTIFACTS_DIR, MODEL_PATHS


BATCH_SIZE = 64
TRAINING_OUTPUT_DIR = ARTIFACTS_DIR / "training" / "fakenewsnet_title_transformer"


def get_latest_checkpoint(training_output_dir: Path = TRAINING_OUTPUT_DIR) -> Path:
    checkpoints = [
        path
        for path in training_output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.split("-")[-1].isdigit()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {training_output_dir}.")
    return max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))


def build_eval_frame() -> pd.DataFrame:
    frame = load_fakenewsnet_titles()
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    labels = frame["label"].map(label_to_id).astype(int)
    _, eval_frame = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return eval_frame.reset_index(drop=True)


def predict_batches(
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

    for start in tqdm(range(0, len(texts), batch_size), desc="Evaluating adapted model"):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_probabilities = torch.softmax(outputs.logits, dim=1).cpu()

        for probability_vector in batch_probabilities:
            probability_list = [float(value) for value in probability_vector.tolist()]
            predicted_index = int(torch.tensor(probability_list).argmax().item())
            predicted_indices.append(predicted_index)
            predicted_labels.append(LABELS[predicted_index])
            probabilities.append(probability_list)

    return predicted_indices, predicted_labels, probabilities


def main() -> None:
    checkpoint_dir = get_latest_checkpoint()
    tokenizer_dir = MODEL_PATHS["fake_news"]["tokenizer"]
    evaluation_frame = build_eval_frame()

    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_dir)
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint_dir)
    predicted_indices, predicted_labels, probabilities = predict_batches(
        model=model,
        tokenizer=tokenizer,
        texts=evaluation_frame["title_text"].tolist(),
    )

    true_labels = evaluation_frame["label"].tolist()
    predictions = build_predictions_frame(
        examples=evaluation_frame,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        predicted_indices=predicted_indices,
        probabilities=probabilities,
        labels=LABELS,
    )
    metrics = compute_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        labels=LABELS,
    )

    output_dir = create_run_directory(
        ARTIFACTS_DIR,
        "domain_adapted",
        "fakenewsnet_titles",
    )
    save_evaluation_outputs(
        output_dir=output_dir,
        predictions=predictions,
        metrics=metrics,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        labels=LABELS,
    )

    print(f"Evaluated checkpoint: {checkpoint_dir}")
    print(f"Saved domain-adapted evaluation artifacts to: {output_dir}")
    print(
        "Accuracy: "
        f"{metrics['accuracy']:.4f}, "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
