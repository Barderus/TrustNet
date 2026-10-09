from functools import lru_cache

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

def _load_model_bundle(model_dir, tokenizer_dir):
    try:
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_dir)
    except OSError as error:
        raise FileNotFoundError(
            "Required model artifacts could not be loaded from "
            f"'{model_dir}' and '{tokenizer_dir}'."
        ) from error
    model.eval()
    return model, tokenizer


@lru_cache(maxsize=1)
def load_fake_news_model():
    return _load_model_bundle(
        "models/fake_news_model/distilbert_fakenews_model",
        "models/fake_news_model/distilbert_fakenews_tokenizer",
    )


@lru_cache(maxsize=1)
def load_stance_model():
    return _load_model_bundle(
        "models/stance_detection_model/distilbert_stanceD",
        "models/stance_detection_model/distilbert_tokenizer_stanceD",
    )
