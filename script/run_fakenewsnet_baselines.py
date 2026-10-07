from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.classical_baselines import (
    create_baseline_run_dir,
    load_fakenewsnet_title_baseline_dataset,
    run_baselines,
)
from utils.project_config import ARTIFACTS_DIR, DATA_DIR


LIMIT = None


def main() -> None:
    dataset = load_fakenewsnet_title_baseline_dataset(DATA_DIR)
    output_dir = create_baseline_run_dir(dataset.task_name, ARTIFACTS_DIR)
    results = run_baselines(dataset, output_dir=output_dir, limit=LIMIT)

    print(f"Saved FakeNewsNet title baseline metrics to: {output_dir / 'metrics.csv'}")
    print(results[["model", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
