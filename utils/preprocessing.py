import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.project_config import DATA_DIR


@dataclass(frozen=True)
class PreprocessingOutputs:
    master_fakenews: Path
    fakenews_preprocessed: Path
    master_stance: Path
    stance_preprocessed: Path


def normalize_text(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    value = value.replace("\n", " ").replace("\r", " ")

    try:
        import contractions

        value = contractions.fix(value)
    except ModuleNotFoundError:
        pass

    value = re.sub(r"http\S+|www\.\S+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def clean_for_model(text: object) -> str:
    value = normalize_text(text).lower()
    value = re.sub("[^a-zA-Z]", " ", value)
    tokens = [
        token
        for token in value.split()
        if token not in ENGLISH_STOP_WORDS and len(token) > 1
    ]
    return " ".join(tokens)


def word_count(text: object) -> int:
    return len(normalize_text(text).split())


def avg_word_length(text: object) -> float:
    words = normalize_text(text).split()
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def sentence_count(text: object) -> int:
    parts = [part for part in re.split(r"[.!?]+", normalize_text(text)) if part.strip()]
    return len(parts)


def lexical_richness(text: object) -> float:
    tokens = clean_for_model(text).split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def punctuation_count(text: object) -> int:
    value = normalize_text(text)
    return sum(1 for char in value if char in string.punctuation)


def punctuation_ratio(text: object) -> float:
    value = normalize_text(text)
    if not value:
        return 0.0
    return punctuation_count(value) / len(value)


def sentiment_scores(text: object) -> tuple[float, float]:
    try:
        from textblob import TextBlob
    except ModuleNotFoundError:
        return 0.0, 0.0

    sentiment = TextBlob(normalize_text(text)).sentiment
    return float(sentiment.polarity), float(sentiment.subjectivity)


def readability_scores(text: object) -> dict[str, float | None]:
    try:
        import textstat
    except ModuleNotFoundError:
        return {
            "flesch_ease": None,
            "flesch_grade": None,
            "gunning_fog": None,
            "smog": None,
            "ari": None,
            "coleman_liau": None,
        }

    value = normalize_text(text)
    try:
        return {
            "flesch_ease": textstat.flesch_reading_ease(value),
            "flesch_grade": textstat.flesch_kincaid_grade(value),
            "gunning_fog": textstat.gunning_fog(value),
            "smog": textstat.smog_index(value),
            "ari": textstat.automated_readability_index(value),
            "coleman_liau": textstat.coleman_liau_index(value),
        }
    except Exception:
        return {
            "flesch_ease": None,
            "flesch_grade": None,
            "gunning_fog": None,
            "smog": None,
            "ari": None,
            "coleman_liau": None,
        }


def interpret_flesch(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 90:
        return "very easy"
    if score >= 70:
        return "easy"
    if score >= 60:
        return "standard"
    if score >= 30:
        return "difficult"
    return "very difficult"


def entity_counts(texts: Iterable[object], enabled: bool = False) -> list[int]:
    values = [normalize_text(text) for text in texts]
    if not enabled:
        return [0] * len(values)

    try:
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
    except Exception:
        return [0] * len(values)

    return [len(doc.ents) for doc in nlp.pipe(values, batch_size=100)]


def build_master_fakenews(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    fake = pd.read_csv(data_dir / "fake-news-kaggle" / "Fake.csv")
    true = pd.read_csv(data_dir / "fake-news-kaggle" / "True.csv")
    fake = fake.assign(real=0)
    true = true.assign(real=1)

    frames = [fake, true]
    more_path = data_dir / "More-fake-news" / "train.tsv"
    if more_path.exists():
        more = pd.read_csv(more_path, sep="\t")
        if "label" in more.columns:
            more = more.rename(columns={"label": "real"})
        frames.append(more)

    frame = pd.concat(frames, ignore_index=True)
    keep_columns = [column for column in ["title", "text", "subject", "date", "real"] if column in frame.columns]
    frame = frame[keep_columns].copy()
    frame = frame.dropna(subset=["title", "text", "real"])
    frame = frame.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)

    frame["clean_title"] = frame["title"].apply(normalize_text)
    frame["clean_text"] = frame["text"].apply(normalize_text)
    frame["text_len"] = frame["clean_text"].str.len()
    frame["avg_word_len"] = frame["clean_text"].apply(avg_word_length)
    frame["num_sents"] = frame["clean_text"].apply(sentence_count)

    sentiments = frame["clean_text"].apply(sentiment_scores)
    frame["polarity"] = sentiments.apply(lambda value: value[0])
    frame["subjectivity"] = sentiments.apply(lambda value: value[1])
    return frame


def build_fakenews_preprocessed(
    master_frame: pd.DataFrame,
    include_entities: bool = False,
) -> pd.DataFrame:
    frame = master_frame.copy()
    frame["prep_text"] = (
        frame["clean_title"].fillna("").astype(str)
        + " "
        + frame["clean_text"].fillna("").astype(str)
    ).apply(clean_for_model)
    frame["word_count"] = frame["prep_text"].apply(word_count)
    frame["unique_words"] = frame["prep_text"].apply(lambda value: len(set(str(value).split())))
    frame["lexical_richness"] = frame["prep_text"].apply(lexical_richness)
    frame["sentence_count"] = frame["clean_text"].apply(sentence_count)

    readability = frame["clean_text"].apply(readability_scores).apply(pd.Series)
    frame = pd.concat([frame, readability], axis=1)
    frame["flesch_interpretation"] = frame["flesch_ease"].apply(interpret_flesch)
    frame["punct_count"] = frame["clean_text"].apply(punctuation_count)
    frame["punct_ratio"] = frame["clean_text"].apply(punctuation_ratio)
    frame["entity_count"] = entity_counts(frame["clean_text"], enabled=include_entities)
    return frame


def _headline_body_overlap(headline: object, body: object) -> float:
    headline_words = set(clean_for_model(headline).split())
    body_words = set(clean_for_model(body).split())
    if not headline_words:
        return 0.0
    return len(headline_words & body_words) / len(headline_words)


def _headline_body_cosine(headline: object, body: object) -> float:
    values = [clean_for_model(headline), clean_for_model(body)]
    if not values[0] or not values[1]:
        return 0.0
    try:
        matrix = TfidfVectorizer().fit_transform(values)
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except ValueError:
        return 0.0


def build_master_stance(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    base_dir = data_dir / "StanceDetection"
    bodies = pd.read_csv(base_dir / "train_bodies.csv")
    stances = pd.read_csv(base_dir / "train_stances.csv")
    frame = stances.merge(bodies, on="Body ID", how="inner")

    frame["headline_len_chars"] = frame["Headline"].fillna("").astype(str).str.len()
    frame["headline_len_words"] = frame["Headline"].apply(word_count)
    frame["body_len_chars"] = frame["articleBody"].fillna("").astype(str).str.len()
    frame["body_len_words"] = frame["articleBody"].apply(word_count)
    frame["headline_body_overlap_ratio"] = frame.apply(
        lambda row: _headline_body_overlap(row["Headline"], row["articleBody"]),
        axis=1,
    )
    frame["cosine_sim"] = frame.apply(
        lambda row: _headline_body_cosine(row["Headline"], row["articleBody"]),
        axis=1,
    )
    frame["headline_sentiment"] = frame["Headline"].apply(lambda value: sentiment_scores(value)[0])
    frame["body_sentiment"] = frame["articleBody"].apply(lambda value: sentiment_scores(value)[0])
    frame["readability"] = frame["articleBody"].apply(lambda value: readability_scores(value)["flesch_ease"])
    return frame


def build_stance_preprocessed(
    master_frame: pd.DataFrame,
    include_entities: bool = False,
) -> pd.DataFrame:
    frame = master_frame.copy()
    frame["headline_prep"] = frame["Headline"].apply(clean_for_model)
    frame["body_prep"] = frame["articleBody"].apply(clean_for_model)
    frame["entity_count"] = entity_counts(frame["articleBody"], enabled=include_entities)
    frame["combined_text"] = (
        frame["headline_prep"].fillna("").astype(str)
        + " [SEP] "
        + frame["body_prep"].fillna("").astype(str)
    ).str.strip()
    frame["stance_label"] = frame["Stance"].fillna("").astype(str).str.upper()
    return frame.dropna(subset=["Headline", "articleBody", "Stance", "headline_prep", "body_prep"])
