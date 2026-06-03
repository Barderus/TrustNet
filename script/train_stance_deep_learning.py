from pathlib import Path

from utils.deep_learning_models import DeepLearningConfig, run_deep_learning_experiments


DATA_PATH = Path(
    r"C:\Users\Barderus_Legion\PycharmProjects\TrustNet\data\preprocessed\stance_preprocessed.csv"
)
OUTPUT_DIR = Path(
    r"C:\Users\Barderus_Legion\PycharmProjects\TrustNet\artifacts\deep_learning\stance"
)

TEXT_COLUMN = "combined_text"
LABEL_COLUMN = "stance_label"
LABEL_NAMES = ["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"]

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
LIMIT = None


def main() -> None:
    config = DeepLearningConfig(
        task_name="stance",
        data_path=DATA_PATH,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        label_names=LABEL_NAMES,
        output_dir=OUTPUT_DIR,
        max_vocab_size=MAX_VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        limit=LIMIT,
    )
    results = run_deep_learning_experiments(config)
    print(f"Saved stance deep-learning results to: {OUTPUT_DIR / 'metrics.csv'}")
    print(results[["model", "accuracy", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
