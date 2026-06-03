import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils.project_config import ARTIFACTS_DIR


@dataclass(frozen=True)
class EvaluationRun:
    task_name: str
    dataset_name: str
    run_dir: Path
    metrics_path: Path
    predictions_path: Path
    confusion_matrix_path: Path
    calibration_path: Path
    summary_path: Path

    def load_metrics(self) -> dict:
        with self.metrics_path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    def load_predictions(self) -> pd.DataFrame:
        return pd.read_csv(self.predictions_path)


def _evaluation_root(base_dir: Path = ARTIFACTS_DIR) -> Path:
    return base_dir / "evaluation"


def list_evaluation_runs(
    task_name: str,
    dataset_name: str | None = None,
    base_dir: Path = ARTIFACTS_DIR,
) -> list[EvaluationRun]:
    task_dir = _evaluation_root(base_dir) / task_name
    if not task_dir.exists():
        return []

    dataset_dirs = [task_dir / dataset_name] if dataset_name else [
        path for path in task_dir.iterdir() if path.is_dir()
    ]

    runs: list[EvaluationRun] = []
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue

        for run_dir in dataset_dir.iterdir():
            if not run_dir.is_dir():
                continue

            metrics_path = run_dir / "metrics.json"
            predictions_path = run_dir / "predictions.csv"
            confusion_matrix_path = run_dir / "confusion_matrix.png"
            calibration_path = run_dir / "calibration.png"
            summary_path = run_dir / "summary.md"

            if not metrics_path.exists() or not predictions_path.exists():
                continue

            runs.append(
                EvaluationRun(
                    task_name=task_name,
                    dataset_name=dataset_dir.name,
                    run_dir=run_dir,
                    metrics_path=metrics_path,
                    predictions_path=predictions_path,
                    confusion_matrix_path=confusion_matrix_path,
                    calibration_path=calibration_path,
                    summary_path=summary_path,
                )
            )

    return sorted(runs, key=lambda run: run.run_dir.name, reverse=True)


def get_latest_evaluation_run(
    task_name: str,
    dataset_name: str | None = None,
    base_dir: Path = ARTIFACTS_DIR,
) -> EvaluationRun:
    runs = list_evaluation_runs(task_name=task_name, dataset_name=dataset_name, base_dir=base_dir)
    if not runs:
        scope = f"{task_name}/{dataset_name}" if dataset_name else task_name
        raise FileNotFoundError(
            f"No evaluation runs were found for '{scope}' under '{_evaluation_root(base_dir)}'."
        )
    return runs[0]


def load_latest_evaluation_artifacts(
    task_name: str,
    dataset_name: str | None = None,
    base_dir: Path = ARTIFACTS_DIR,
) -> tuple[dict, pd.DataFrame, EvaluationRun]:
    run = get_latest_evaluation_run(task_name=task_name, dataset_name=dataset_name, base_dir=base_dir)
    metrics = run.load_metrics()
    predictions = run.load_predictions()
    return metrics, predictions, run
