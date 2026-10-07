"""Report text and body overlap in the current fake-news and stance splits.

Run from the repository root with: python -m script.audit_split_overlap
"""

from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.datasets import load_fake_news_kaggle_bundle, load_stance_detection_bundle
from utils.preprocessing import clean_for_model
from utils.project_config import ARTIFACTS_DIR
from utils.transformer_training import fake_news_training_config, stance_training_config


def near_duplicate_eval_texts(train_texts, eval_texts):
    """Screen for very similar long texts sharing their first 20 cleaned tokens."""
    train_by_prefix = defaultdict(list)
    train_keys = set(train_texts)
    for text in train_keys:
        words = text.split()
        if len(words) >= 40:
            train_by_prefix[tuple(words[:20])].append((set(words), len(words)))

    near_texts = set()
    for text in set(eval_texts) - train_keys:
        words = text.split()
        if len(words) < 40:
            continue
        tokens = set(words)
        for other_tokens, other_length in train_by_prefix.get(tuple(words[:20]), ()):
            if abs(len(words) - other_length) > max(len(words), other_length) * 0.1:
                continue
            if len(tokens & other_tokens) / len(tokens | other_tokens) >= 0.9:
                near_texts.add(text)
                break
    return near_texts


def audit_fake_news():
    config = fake_news_training_config(ARTIFACTS_DIR)
    frame = pd.read_csv(config.data_path, usecols=["prep_text", "real"])
    frame = frame.dropna(subset=["prep_text", "real"]).copy()
    train_indices, eval_indices = train_test_split(
        frame.index,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=frame["real"].astype(int),
    )
    train_texts = frame.loc[train_indices, "prep_text"].astype(str)
    eval_texts = frame.loc[eval_indices, "prep_text"].astype(str)
    train_keys = set(train_texts)
    near_keys = near_duplicate_eval_texts(train_texts, eval_texts)

    benchmark = load_fake_news_kaggle_bundle().test
    print("Fake-news transformer validation:")
    print(f"  stratified split: test_size={config.test_size}, seed={config.random_state}")
    print(f"  rows: {len(eval_texts)}")
    print(f"  training label counts: {frame.loc[train_indices, 'real'].value_counts().to_dict()}")
    print(f"  validation label counts: {frame.loc[eval_indices, 'real'].value_counts().to_dict()}")
    print(f"  exact cleaned-text matches in training: {eval_texts.isin(train_keys).sum()}")
    print(f"  additional near-duplicate screen matches: {eval_texts.isin(near_keys).sum()}")
    print("Kaggle benchmark holdout:")
    print(f"  rows: {len(benchmark)}")
    print(
        "  exact cleaned-text matches in transformer training: "
        f"{benchmark['model_input'].isin(train_keys).sum()}"
    )


def audit_stance():
    config = stance_training_config(ARTIFACTS_DIR)
    frame = pd.read_csv(config.data_path, usecols=["Body ID", "combined_text", "stance_label"])
    frame = frame.dropna(subset=["combined_text", "stance_label"]).copy()
    train_indices, eval_indices = train_test_split(
        frame.index,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=frame["stance_label"].astype(str).str.upper(),
    )
    train_bodies = set(frame.loc[train_indices, "Body ID"])
    eval_bodies = frame.loc[eval_indices, "Body ID"]

    bundle = load_stance_detection_bundle()
    development_bodies = bundle.train.drop_duplicates(subset=["Body ID"])
    competition_bodies = bundle.test.drop_duplicates(subset=["Body ID"]).copy()
    development_ids = set(development_bodies["Body ID"])
    development_texts = set(development_bodies["body"].apply(clean_for_model))
    competition_bodies["clean_body"] = competition_bodies["body"].apply(clean_for_model)
    shared_text_ids = set(
        competition_bodies.loc[
            competition_bodies["clean_body"].isin(development_texts), "Body ID"
        ]
    )

    print("Stance transformer validation:")
    print(f"  stratified split: test_size={config.test_size}, seed={config.random_state}")
    print(f"  rows: {len(eval_bodies)}")
    print(f"  training label counts: {frame.loc[train_indices, 'stance_label'].value_counts().to_dict()}")
    print(f"  validation label counts: {frame.loc[eval_indices, 'stance_label'].value_counts().to_dict()}")
    print(f"  rows sharing Body ID with training: {eval_bodies.isin(train_bodies).sum()}")
    print("FNC-1 competition set:")
    print(f"  rows: {len(bundle.test)}")
    print(
        "  Body IDs shared with development: "
        f"{len(development_ids & set(competition_bodies['Body ID']))}"
    )
    print(f"  bodies with identical cleaned development text: {len(shared_text_ids)}")
    print(f"  stance rows on those bodies: {bundle.test['Body ID'].isin(shared_text_ids).sum()}")


def main():
    audit_fake_news()
    audit_stance()
    print("Near-duplicate screen: first 20 cleaned tokens match, at least 40 tokens,")
    print("length within 10%, and token-set Jaccard at least 0.90; not exhaustive.")


if __name__ == "__main__":
    main()
