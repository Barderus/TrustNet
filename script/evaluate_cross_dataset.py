import pandas as pd
import torch
from tqdm import tqdm

from utils.evaluation import (
    build_predictions_frame,
    compute_metrics,
)
from utils.model_loader import load_fake_news_model
from utils.preprocessing import clean_for_model


LIMIT = None
BATCH_SIZE = 64
LABELS = ["FAKE", "REAL"]


def load_fakenewsnet_titles():
    frames = []
    for file_name, label in [
        ("gossipcop_fake.csv", "FAKE"),
        ("politifact_fake.csv", "FAKE"),
        ("gossipcop_real.csv", "REAL"),
        ("politifact_real.csv", "REAL"),
    ]:
        path = f"data/FakeNewsNet/{file_name}"
        frame = pd.read_csv(path)
        frame = frame.assign(label=label, source_file=file_name)
        frames.append(frame)

    full_frame = pd.concat(frames, ignore_index=True)
    full_frame["input_text"] = full_frame["title"].fillna("").astype(str).str.strip()
    full_frame = full_frame.loc[full_frame["input_text"].str.len() > 0].copy()
    full_frame["model_input"] = full_frame["input_text"].apply(clean_for_model)
    return full_frame.reset_index(drop=True)


def predict_batches(
    model,
    tokenizer,
    texts,
    batch_size=BATCH_SIZE,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predicted_indices = []
    predicted_labels = []
    probabilities = []

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


def main():
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
        texts=evaluation_frame["model_input"].tolist(),
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

    print(
        "Accuracy: "
        f"{metrics['accuracy']:.4f}, "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )
    return predictions, metrics


if __name__ == "__main__":
    main()
