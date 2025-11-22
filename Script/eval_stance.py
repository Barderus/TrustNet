import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import shap
import webbrowser
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*_register_pytree_node.*",
    category=FutureWarning
)


# -----------------------------
# LOAD MODEL + TOKENIZER
# -----------------------------
MODEL_PATH = "../Notebooks/distilbert_stanceD"
TOKENIZER_PATH = "../Notebooks/distilbert_tokenizer_stanceD"

print("Loading model...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)

model.eval()

LABELS = ["agree", "disagree", "discuss", "unrelated"]

HEADLINE = "Axios reported that Kamala Harris pledged to “spend hundreds of millions of dollars on the wall along the southern border."
BODY = "In fact, Harris said nothing of the sort, but Axios inferred this based on her support of a bill that makes funds “available” for a border barrier but doesn’t actually require it to be built. Such funds were appropriated in prior years but were not used by the Biden administration."


# -----------------------------
# CLEAN SHAP PREDICTION FN
# -----------------------------
def predict_fn(texts):
    clean_texts = []

    for t in texts:
        if not isinstance(t, str):
            if hasattr(t, "__array__"):
                t = t.tolist()
            if isinstance(t, list):
                t = "".join(str(x) for x in t)

        clean_texts.append(str(t))

    inputs = tokenizer(
        clean_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    return probs.cpu().numpy()


# -----------------------------
# STANCE PREDICTION (normal)
# -----------------------------
def predict_stance(headline, body):
    combined_text = headline.strip() + " [SEP] " + body.strip()

    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_index = probs.argmax()
    pred_label = LABELS[pred_index]

    return combined_text, pred_label, probs


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print("\n TrustNet — Stance Detection with SHAP\n")

    #headline = input("Enter HEADLINE:\n> ")
    #body = input("\nEnter BODY:\n> ")

    combined_text, predicted_label, probabilities = predict_stance(HEADLINE, BODY)

    # ---- OUTPUT PREDICTION ----
    print("\n==============================")
    print(f"Prediction: {predicted_label.upper()}")
    print("==============================")
    for idx, label in enumerate(LABELS):
        print(f"{label:10s}: {probabilities[idx]:.4f}")

    # ---- SHAP EXPLANATION ----
    print("\nGenerating SHAP explanation...")

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_fn, masker)

    shap_values = explainer([combined_text])

    # ---- SAVE AS HTML ----
    html_obj = shap.plots.text(shap_values[0], display=False)

    # SHAP sometimes returns a Visualizer, sometimes a raw HTML string
    html_output = html_obj.data if hasattr(html_obj, "data") else html_obj

    output_file = "stance_shap_explanation.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"\nSHAP explanation saved to: {output_file}")
    webbrowser.open(output_file)

