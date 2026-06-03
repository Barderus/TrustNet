from pathlib import Path
import warnings

import shap
import torch

from utils.model_loader import load_stance_model
from utils.prediction import predict_text


HEADLINE = "Enter a headline or claim here."
BODY = "Enter the article body here."
SAVE_SHAP = False
OUTPUT_DIR = Path(r"C:\Users\Barderus_Legion\PycharmProjects\TrustNet\artifacts\explainability")
LABELS = ["agree", "disagree", "discuss", "unrelated"]


warnings.filterwarnings(
    "ignore",
    message=".*_register_pytree_node.*",
    category=FutureWarning,
)


def build_stance_input(headline: str, body: str) -> str:
    return f"{headline.strip()} [SEP] {body.strip()}"


def make_shap_predict_fn(model, tokenizer):
    def predict_fn(texts):
        clean_texts = []
        for text in texts:
            if not isinstance(text, str):
                if hasattr(text, "__array__"):
                    text = text.tolist()
                if isinstance(text, list):
                    text = "".join(str(value) for value in text)
            clean_texts.append(str(text))

        inputs = tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        return probabilities.cpu().numpy()

    return predict_fn


def save_shap_explanation(model, tokenizer, combined_text: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(make_shap_predict_fn(model, tokenizer), masker)
    shap_values = explainer([combined_text])
    html_object = shap.plots.text(shap_values[0], display=False)
    html_output = html_object.data if hasattr(html_object, "data") else html_object

    output_path = output_dir / "stance_shap_explanation.html"
    output_path.write_text(html_output, encoding="utf-8")
    return output_path


def main() -> None:
    model, tokenizer = load_stance_model()
    model.eval()

    combined_text = build_stance_input(HEADLINE, BODY)
    predicted_index, probabilities, _ = predict_text(model, tokenizer, combined_text)
    predicted_label = LABELS[predicted_index]

    print(f"Prediction: {predicted_label.upper()}")
    for label, probability in zip(LABELS, probabilities):
        print(f"{label:10s}: {float(probability):.4f}")

    if SAVE_SHAP:
        output_path = save_shap_explanation(
            model=model,
            tokenizer=tokenizer,
            combined_text=combined_text,
            output_dir=OUTPUT_DIR,
        )
        print(f"SHAP explanation saved to: {output_path}")


if __name__ == "__main__":
    main()
