from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from utils.datasets import GROUPED_SPLIT_VERSION


def require_grouped_split_model(model):
    if getattr(model.config, "trustnet_split_version", None) != GROUPED_SPLIT_VERSION:
        raise ValueError(
            "This saved model predates the grouped evaluation split. "
            "Retrain the transformer before generating benchmark results."
        )


def normalize_probability_rows(probabilities):
    normalized = []
    for probability_vector in probabilities:
        values = [float(value) for value in probability_vector]
        total = sum(values)
        if total <= 0:
            normalized.append(values)
        else:
            normalized.append([value / total for value in values])
    return normalized


def build_predictions_frame(
    examples,
    true_labels,
    predicted_labels,
    predicted_indices,
    probabilities,
    labels,
):
    predictions = examples.copy().reset_index(drop=True)
    predictions["true_label"] = true_labels
    predictions["predicted_label"] = predicted_labels
    predictions["predicted_index"] = predicted_indices

    for label_index, label_name in enumerate(labels):
        predictions[f"prob_{label_name.lower()}"] = [
            float(prob_vector[label_index]) for prob_vector in probabilities
        ]

    predictions["confidence"] = [
        float(max(prob_vector)) for prob_vector in probabilities
    ]
    predictions["is_correct"] = (
        predictions["true_label"] == predictions["predicted_label"]
    )
    return predictions


def compute_metrics(
    true_labels,
    predicted_labels,
    probabilities,
    labels,
):
    probabilities = normalize_probability_rows(probabilities)
    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
    }

    try:
        if len(labels) == 2:
            positive_label_probabilities = [
                probability_vector[1] for probability_vector in probabilities
            ]
            metrics["roc_auc"] = float(
                roc_auc_score(true_labels, positive_label_probabilities)
            )
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(
                    true_labels,
                    probabilities,
                    labels=labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["log_loss"] = float(log_loss(true_labels, probabilities, labels=labels))
    except ValueError:
        metrics["log_loss"] = None

    return metrics
