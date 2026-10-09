from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def baseline_models(random_state=42):
    def vectorizer():
        return TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
        )

    return {
        "logistic_regression_tfidf": Pipeline(
            [
                ("tfidf", vectorizer()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        "linear_svc_tfidf": Pipeline(
            [
                ("tfidf", vectorizer()),
                ("model", LinearSVC(random_state=random_state)),
            ]
        ),
        "ridge_classifier_tfidf": Pipeline(
            [
                ("tfidf", vectorizer()),
                ("model", RidgeClassifier()),
            ]
        ),
    }


def compute_classification_metrics(
    y_true,
    y_pred,
    labels,
    y_score=None,
):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    metrics["roc_auc"] = calculate_roc_auc(y_true, y_score, labels)
    return metrics


def calculate_roc_auc(y_true, y_score, labels):
    if y_score is None:
        return None

    try:
        if len(labels) == 2:
            if len(y_score.shape) == 1:
                return float(roc_auc_score(y_true, y_score))
            return float(roc_auc_score(y_true, y_score[:, 1]))

        return float(
            roc_auc_score(
                y_true,
                y_score,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return None
