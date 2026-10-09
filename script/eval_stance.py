from utils.model_loader import load_stance_model
from utils.prediction import predict_text
from utils.preprocessing import prepare_stance_input


HEADLINE = "Enter a headline or claim here."
BODY = "Enter the article body here."
LABELS = ["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"]


def main():
    model, tokenizer = load_stance_model()
    model_input = prepare_stance_input(HEADLINE, BODY)
    predicted_index, probabilities, _ = predict_text(
        model, tokenizer, model_input
    )

    print("Prediction:", LABELS[int(predicted_index)])
    for label, probability in zip(LABELS, probabilities):
        print(f"{label:10s}: {float(probability):.4f}")


if __name__ == "__main__":
    main()
