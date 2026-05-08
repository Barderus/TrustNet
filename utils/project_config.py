from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("TRUSTNET_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("TRUSTNET_MODELS_DIR", PROJECT_ROOT / "Models"))
ARTIFACTS_DIR = Path(os.getenv("TRUSTNET_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
DOCS_DIR = PROJECT_ROOT / "docs"


MODEL_PATHS = {
    "fake_news": {
        "model": MODELS_DIR / "fake_news_model" / "distilbert_fakenews_model",
        "tokenizer": MODELS_DIR / "fake_news_model" / "distilbert_fakenews_tokenizer",
    },
    "stance": {
        "model": MODELS_DIR / "stance_detection_model" / "distilbert_stanceD",
        "tokenizer": MODELS_DIR / "stance_detection_model" / "distilbert_tokenizer_stanceD",
    },
}


def ensure_project_root_on_path() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def get_model_paths(model_key: str) -> tuple[Path, Path]:
    try:
        paths = MODEL_PATHS[model_key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_PATHS))
        raise ValueError(
            f"Unknown model key '{model_key}'. Available model keys: {available}."
        ) from exc

    return paths["model"], paths["tokenizer"]
