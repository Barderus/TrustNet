import json
from datetime import UTC, datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_loader import load_fake_news_model
from utils.prediction import (
    clean_special_tokens,
    get_word_attributions,
    merge_wordpiece_tokens,
    top_k_attributions,
)
from utils.project_config import ARTIFACTS_DIR


PREDICTIONS_PATH = (
    ARTIFACTS_DIR
    / "evaluation"
    / "cross_dataset"
    / "fakenewsnet_titles"
    / "20261009T151729Z"
    / "predictions.csv"
)
EXAMPLE_ROWS = [5481, 8066, 16255, 2, 5755, 5759]
LOW_CONFIDENCE_ROW = 22444


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    model, tokenizer = load_fake_news_model()
    examples = []
    for row_index in EXAMPLE_ROWS:
        row = predictions.loc[row_index]
        attributions = get_word_attributions(
            model,
            tokenizer,
            row["model_input"],
            class_index=int(row["predicted_index"]),
        )
        terms = top_k_attributions(
            clean_special_tokens(merge_wordpiece_tokens(attributions)), k=8
        )
        examples.append(
            {
                "prediction_row": row_index,
                "dataset": "FakeNewsNet titles",
                "source_file": row["source_file"],
                "title": row["title"],
                "model_input": row["model_input"],
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "prob_fake": float(row["prob_fake"]),
                "prob_real": float(row["prob_real"]),
                "confidence": float(row["confidence"]),
                "top_terms_for_predicted_class": terms,
            }
        )
        print(
            f"{row_index}: {row['true_label']} -> {row['predicted_label']}, "
            f"top terms: {terms[:4]}"
        )

    uncertain = predictions.loc[LOW_CONFIDENCE_ROW]
    output = {
        "source_predictions": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
        "explanation_method": "transformers-interpret token attribution for the predicted class on cleaned title input",
        "examples": examples,
        "low_confidence_example": {
            "prediction_row": LOW_CONFIDENCE_ROW,
            "source_file": uncertain["source_file"],
            "title": uncertain["title"],
            "true_label": uncertain["true_label"],
            "predicted_label": uncertain["predicted_label"],
            "confidence": float(uncertain["confidence"]),
        },
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        ARTIFACTS_DIR
        / "explainability"
        / "cross_dataset"
        / "fakenewsnet_titles"
        / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "examples.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved reviewed examples to: {output_dir}")


if __name__ == "__main__":
    main()
