import json
from pathlib import Path
import re
import sys

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.datasets import load_fake_news_kaggle_bundle
from utils.evaluation import create_run_directory, require_grouped_split_model
from utils.model_loader import load_fake_news_model
from utils.prediction import predict_text
from utils.project_config import ARTIFACTS_DIR


BENCHMARK_PREDICTIONS = (
    ARTIFACTS_DIR
    / "evaluation"
    / "fake_news"
    / "fake-news-kaggle"
    / "20261009T144412Z"
    / "predictions.csv"
)
REAL_SAMPLE_SIZE = 200
RANDOM_STATE = 42
REUTERS_PATTERN = re.compile(r"\breuters\b", re.IGNORECASE)


def main() -> None:
    bundle = load_fake_news_kaggle_bundle()
    saved = pd.read_csv(BENCHMARK_PREDICTIONS)
    if len(saved) != len(bundle.test) or not saved["model_input"].fillna("").equals(
        bundle.test["model_input"].fillna("")
    ):
        raise ValueError("Saved benchmark predictions do not match the current holdout")

    cue_present = saved["model_input"].str.contains(REUTERS_PATTERN)
    real_indices = saved.index[cue_present & saved["true_label"].eq("REAL")]
    fake_indices = saved.index[cue_present & saved["true_label"].eq("FAKE")]
    selected_real = pd.Series(real_indices).sample(
        n=min(REAL_SAMPLE_SIZE, len(real_indices)), random_state=RANDOM_STATE
    )
    selected_indices = sorted([*selected_real.tolist(), *fake_indices.tolist()])

    model, tokenizer = load_fake_news_model()
    require_grouped_split_model(model)
    rows = []
    for row_index in tqdm(selected_indices, desc="Removing Reuters cue"):
        original = saved.loc[row_index]
        changed_text = REUTERS_PATTERN.sub("", original["model_input"])
        changed_text = " ".join(changed_text.split())
        prediction_index, probabilities, _ = predict_text(
            model, tokenizer, changed_text
        )
        changed_label = bundle.labels[int(prediction_index)]
        rows.append(
            {
                "benchmark_row": row_index,
                "true_label": original["true_label"],
                "original_label": original["predicted_label"],
                "ablated_label": changed_label,
                "original_prob_real": float(original["prob_real"]),
                "ablated_prob_real": float(probabilities[1]),
                "cue_occurrences": len(REUTERS_PATTERN.findall(original["model_input"])),
                "label_changed": changed_label != original["predicted_label"],
            }
        )

    results = pd.DataFrame(rows)
    summary = {
        "source_predictions": str(BENCHMARK_PREDICTIONS.relative_to(PROJECT_ROOT)),
        "removed_token": "reuters",
        "random_state": RANDOM_STATE,
        "real_sample_size": REAL_SAMPLE_SIZE,
        "sample_size": len(results),
        "label_changes": int(results["label_changed"].sum()),
        "by_true_label": {},
    }
    for label, group in results.groupby("true_label"):
        summary["by_true_label"][label] = {
            "examples": len(group),
            "label_changes": int(group["label_changed"].sum()),
            "original_accuracy": float((group["original_label"] == label).mean()),
            "ablated_accuracy": float((group["ablated_label"] == label).mean()),
            "mean_change_prob_real": float(
                (group["ablated_prob_real"] - group["original_prob_real"]).mean()
            ),
        }

    output_dir = create_run_directory(
        ARTIFACTS_DIR, "source_cue_ablation", bundle.name
    )
    results.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved source-cue diagnostic to: {output_dir}")


if __name__ == "__main__":
    main()
