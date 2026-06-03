from pathlib import Path

from utils.classical_baselines import (
    create_baseline_run_dir,
    load_fake_news_baseline_dataset,
    load_stance_baseline_dataset,
    run_baselines,
)


TASK = "all"
LIMIT = None
DATA_DIR = Path(r"C:\Users\Barderus_Legion\PycharmProjects\TrustNet\data")
ARTIFACTS_DIR = Path(r"C:\Users\Barderus_Legion\PycharmProjects\TrustNet\artifacts")


def run_task(task_name: str) -> None:
    if task_name == "fake_news":
        dataset = load_fake_news_baseline_dataset(DATA_DIR)
    elif task_name == "stance":
        dataset = load_stance_baseline_dataset(DATA_DIR)
    else:
        raise ValueError(f"Unknown baseline task: {task_name}")

    output_dir = create_baseline_run_dir(dataset.task_name, ARTIFACTS_DIR)
    results = run_baselines(dataset, output_dir=output_dir, limit=LIMIT)
    print(f"Saved {task_name} baseline metrics to: {output_dir / 'metrics.csv'}")
    print(results[["model", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))


def main() -> None:
    if TASK in {"fake_news", "all"}:
        run_task("fake_news")

    if TASK in {"stance", "all"}:
        run_task("stance")


if __name__ == "__main__":
    main()
