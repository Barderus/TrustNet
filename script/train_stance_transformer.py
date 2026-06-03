from pathlib import Path

from utils.transformer_training import TransformerTrainingConfig, train_transformer


DATA_PATH = Path(
    r"C:\Users\Barderus_Legion\PycharmProjects\TrustNet\data\preprocessed\stance_preprocessed.csv"
)
MODEL_OUTPUT_DIR = Path(
    r"C:\Users\Barderus_Legion\PycharmProjects\TrustNet\models\stance_detection_model\distilbert_stanceD"
)
TOKENIZER_OUTPUT_DIR = Path(
    r"C:\Users\Barderus_Legion\PycharmProjects\TrustNet\models\stance_detection_model\distilbert_tokenizer_stanceD"
)
TRAINING_OUTPUT_DIR = Path(
    r"C:\Users\Barderus_Legion\PycharmProjects\TrustNet\artifacts\training\stance_transformer"
)

TEXT_COLUMN = "combined_text"
LABEL_COLUMN = "stance_label"
LABEL_NAMES = ["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"]

EPOCHS = 3.0
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
LIMIT = None


def main() -> None:
    config = TransformerTrainingConfig(
        task_name="stance",
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
    print(f"Saved stance transformer model to: {model_dir}")


if __name__ == "__main__":
    main()
