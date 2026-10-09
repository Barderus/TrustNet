"""Report text and body overlap in the current fake-news and stance splits.

Run from the repository root with: python -m script.audit_split_overlap
"""

from utils.datasets import (
    load_fake_news_kaggle_bundle,
    load_stance_detection_bundle,
    near_duplicate_eval_texts,
)
from utils.preprocessing import clean_for_model
from utils.transformer_training import prepare_training_split


TEST_SIZE = 0.2
RANDOM_STATE = 42


def audit_fake_news():
    train_frame, eval_frame = prepare_training_split(
        task_name="fake_news",
        data_path="data/preprocessed/fakenews_preprocessed.csv",
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    benchmark = load_fake_news_kaggle_bundle().test
    train_texts = train_frame["prep_text"].astype(str)
    eval_texts = eval_frame["prep_text"].astype(str)
    train_keys = set(train_texts)
    benchmark_keys = set(benchmark["model_input"])
    development_texts = train_texts.tolist() + eval_texts.tolist()
    near_validation = near_duplicate_eval_texts(train_texts, eval_texts)
    near_benchmark = near_duplicate_eval_texts(development_texts, benchmark_keys)
    print("Fake-news transformer validation:")
    print(f"  grouped split: test_size={TEST_SIZE}, seed={RANDOM_STATE}")
    print(f"  rows: {len(eval_texts)}")
    print(f"  training label counts: {train_frame['real'].value_counts().to_dict()}")
    print(f"  validation label counts: {eval_frame['real'].value_counts().to_dict()}")
    print(
        "  shared split groups: "
        f"{eval_frame['split_group'].isin(set(train_frame['split_group'])).sum()}"
    )
    print(f"  exact cleaned-text matches in training: {eval_texts.isin(train_keys).sum()}")
    print(f"  additional near-duplicate screen matches: {eval_texts.isin(near_validation).sum()}")
    print("Kaggle benchmark holdout:")
    print(f"  rows: {len(benchmark)}")
    print(
        "  groups shared with training: "
        f"{benchmark['split_group'].isin(set(train_frame['split_group'])).sum()}"
    )
    print(
        "  groups shared with validation: "
        f"{benchmark['split_group'].isin(set(eval_frame['split_group'])).sum()}"
    )
    print(
        "  exact cleaned-text matches in transformer development: "
        f"{benchmark['model_input'].isin(train_keys | set(eval_texts)).sum()}"
    )
    print(
        "  additional near-duplicate screen matches in development: "
        f"{benchmark['model_input'].isin(near_benchmark).sum()}"
    )


def audit_stance():
    train_frame, eval_frame = prepare_training_split(
        task_name="stance",
        data_path="data/preprocessed/stance_preprocessed.csv",
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
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
    print(f"  grouped split: test_size={TEST_SIZE}, seed={RANDOM_STATE}")
    print(f"  rows: {len(eval_frame)}")
    print(f"  training label counts: {train_frame['stance_label'].value_counts().to_dict()}")
    print(f"  validation label counts: {eval_frame['stance_label'].value_counts().to_dict()}")
    print(
        "  groups shared with training: "
        f"{eval_frame['split_group'].isin(set(train_frame['split_group'])).sum()}"
    )
    print(
        "  rows sharing Body ID with training: "
        f"{eval_frame['Body ID'].isin(set(train_frame['Body ID'])).sum()}"
    )
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
