from pathlib import Path

import pandas as pd

from utils.preprocessing import (
    build_fakenews_preprocessed,
    build_master_fakenews,
    build_master_stance,
    build_stance_preprocessed,
)


TASK = "all"
STAGE = "all"
INCLUDE_ENTITIES = False
DATA_DIR = Path(r"C:\Users\Barderus_Legion\PycharmProjects\TrustNet\data")
PREPROCESSED_DIR = DATA_DIR / "preprocessed"


def run_fake_news() -> None:
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master_path = PREPROCESSED_DIR / "master_fakenews.csv"
    model_path = PREPROCESSED_DIR / "fakenews_preprocessed.csv"

    if STAGE in {"master", "all"} or not master_path.exists():
        master = build_master_fakenews(DATA_DIR)
        master.to_csv(master_path, index=False)
        print(f"Saved fake-news master data to: {master_path}")
    else:
        master = pd.read_csv(master_path)

    if STAGE in {"model", "all"}:
        preprocessed = build_fakenews_preprocessed(
            master,
            include_entities=INCLUDE_ENTITIES,
        )
        preprocessed.to_csv(model_path, index=False)
        print(f"Saved fake-news model data to: {model_path}")


def run_stance() -> None:
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master_path = PREPROCESSED_DIR / "master_stance.csv"
    model_path = PREPROCESSED_DIR / "stance_preprocessed.csv"

    if STAGE in {"master", "all"} or not master_path.exists():
        master = build_master_stance(DATA_DIR)
        master.to_csv(master_path, index=False)
        print(f"Saved stance master data to: {master_path}")
    else:
        master = pd.read_csv(master_path)

    if STAGE in {"model", "all"}:
        preprocessed = build_stance_preprocessed(
            master,
            include_entities=INCLUDE_ENTITIES,
        )
        preprocessed.to_csv(model_path, index=False)
        print(f"Saved stance model data to: {model_path}")


def main() -> None:
    if TASK in {"fake_news", "all"}:
        run_fake_news()

    if TASK in {"stance", "all"}:
        run_stance()


if __name__ == "__main__":
    main()
