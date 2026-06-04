from pathlib import Path
import sys

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.evaluation import (
    build_predictions_frame,
    compute_metrics,
    create_run_directory,
    save_evaluation_outputs,
)
from utils.model_loader import load_fake_news_model
from utils.project_config import ARTIFACTS_DIR, DATA_DIR


DATASET = "fakenewsnet_titles"
LIMIT = None
BATCH_SIZE = 64
LABELS = ["FAKE", "REAL"]


def load_fakenewsnet_titles(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    base_dir = data_dir / "FakeNewsNet"
    frames = []
    for file_name, label in [
        ("gossipcop_fake.csv", "FAKE"),
        ("politifact_fake.csv", "FAKE"),
        ("gossipcop_real.csv", "REAL"),
        ("politifact_real.csv", "REAL"),
    ]:
        path = base_dir / file_name
        frame = pd.read_csv(path)
        frame = frame.assign(label=label, source_file=file_name)
        frames.append(frame)

    full_frame = pd.concat(frames, ignore_index=True)
    full_frame["input_text"] = full_frame["title"].fillna("").astype(str).str.strip()
    full_frame = full_frame.loc[full_frame["input_text"].str.len() > 0].copy()
    return full_frame.reset_index(drop=True)


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

    for start in tqdm(range(0, len(texts), batch_size), desc="Evaluating FakeNewsNet"):
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
            predicted_labels.append(LABELS[predicted_index])
            probabilities.append(probability_list)

    return predicted_indices, predicted_labels, probabilities


def main() -> None:
    evaluation_frame = load_fakenewsnet_titles()
    if LIMIT is not None:
        evaluation_frame = evaluation_frame.sample(
            n=min(LIMIT, len(evaluation_frame)),
            random_state=42,
        ).reset_index(drop=True)

    model, tokenizer = load_fake_news_model()
    predicted_indices, predicted_labels, probabilities = predict_batches(
        model=model,
        tokenizer=tokenizer,
        texts=evaluation_frame["input_text"].tolist(),
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

    output_dir = create_run_directory(ARTIFACTS_DIR, "cross_dataset", DATASET)
    save_evaluation_outputs(
        output_dir=output_dir,
        predictions=predictions,
        metrics=metrics,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        labels=LABELS,
    )

    print(f"Saved cross-dataset evaluation artifacts to: {output_dir}")
    print(
        "Accuracy: "
        f"{metrics['accuracy']:.4f}, "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
