import torch

def run_prediction(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).flatten().tolist()
        pred = torch.argmax(logits, dim=1).item()

    return pred, probs
