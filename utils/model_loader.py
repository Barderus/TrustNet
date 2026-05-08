from functools import lru_cache

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from utils.project_config import get_model_paths

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


def _load_model_bundle(model_key):
    model_path, tokenizer_path = get_model_paths(model_key)

    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            "Required model artifacts are missing. "
            f"Expected model at '{model_path}' and tokenizer at '{tokenizer_path}'. "
            "Set TRUSTNET_MODELS_DIR to your local model artifact directory or add the "
            "expected folders under the repository's Models/ directory."
        )

    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


@lru_cache(maxsize=1)
def load_fake_news_model():
    return _load_model_bundle("fake_news")


@lru_cache(maxsize=1)
def load_stance_model():
    return _load_model_bundle("stance")


if st is not None:
    load_fake_news_model = st.cache_resource(load_fake_news_model)
    load_stance_model = st.cache_resource(load_stance_model)
