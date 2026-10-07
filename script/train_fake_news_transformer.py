from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.transformer_training import TransformerTrainingConfig, train_transformer
from utils.project_config import ARTIFACTS_DIR, DATA_DIR, MODEL_PATHS


DATA_PATH = DATA_DIR / "preprocessed" / "fakenews_preprocessed.csv"
MODEL_OUTPUT_DIR = MODEL_PATHS["fake_news"]["model"]
TOKENIZER_OUTPUT_DIR = MODEL_PATHS["fake_news"]["tokenizer"]
TRAINING_OUTPUT_DIR = ARTIFACTS_DIR / "training" / "fake_news_transformer"

TEXT_COLUMN = "prep_text"
LABEL_COLUMN = "real"
LABEL_NAMES = ["FAKE", "REAL"]

EPOCHS = 3.0
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
LIMIT = None


def main() -> None:
    config = TransformerTrainingConfig(
        task_name="fake_news",
        data_path=DATA_PATH,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        label_names=LABEL_NAMES,
        model_output_dir=MODEL_OUTPUT_DIR,
        tokenizer_output_dir=TOKENIZER_OUTPUT_DIR,
        training_output_dir=TRAINING_OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        limit=LIMIT,
    )
    model_dir = train_transformer(config)
    print(f"Saved fake_news transformer model to: {model_dir}")


if __name__ == "__main__":
    main()
