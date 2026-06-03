import torch

def chunk_text(tokenizer, text, max_length=512, stride=50):
    """
    Splits text into overlapping token chunks.
    Returns a list of token ID tensors.
    """
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        verbose=False,
    )["input_ids"][0]

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_length
        chunk_ids = tokens[start:end]
        chunks.append(chunk_ids)
        start += max_length - stride  # overlap preserves context

    return chunks


def run_prediction(model, tokenizer, text, temperature=1.0):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, num_labels]
        probs = torch.softmax(logits / temperature, dim=1)[0].cpu().numpy()

    pred = probs.argmax()

    return pred, probs


def run_prediction_chunked(model, tokenizer, text, temperature=1.0):
    chunks = chunk_text(tokenizer, text)

    all_probs = []

    with torch.no_grad():
        for chunk in chunks:
            inputs = {
                "input_ids": chunk.unsqueeze(0),
                "attention_mask": torch.ones_like(chunk).unsqueeze(0)
            }

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits / temperature, dim=1)
            all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0)        # [num_chunks, num_labels]
    mean_probs = torch.mean(all_probs, dim=0)      # aggregate
    pred = mean_probs.argmax().item()

    # Identify most influential chunk (for explainability)
    best_chunk_idx = all_probs.max(dim=1).values.argmax().item()

    return pred, mean_probs.cpu().numpy(), best_chunk_idx


def predict_text(model, tokenizer, text, temperature=1.0):
    token_count = len(
        tokenizer(
            text,
            truncation=False,
            add_special_tokens=True,
            verbose=False,
        )["input_ids"]
    )

    if token_count <= 512:
        pred, probs = run_prediction(model, tokenizer, text, temperature)
        return pred, probs, text
    else:
        pred, probs, best_idx = run_prediction_chunked(
            model, tokenizer, text, temperature
        )

        chunks = chunk_text(tokenizer, text)
        best_chunk = chunks[best_idx]
        best_text = tokenizer.decode(best_chunk, skip_special_tokens=True)

        return pred, probs, best_text



def get_word_attributions(model, tokenizer, text):
    from transformers_interpret import SequenceClassificationExplainer

    explainer = SequenceClassificationExplainer(model, tokenizer)

    try:
        # This returns a list of tuples: (token, attribution)
        attributions = explainer(text)
    except Exception as e:
        return [("ERROR", str(e))]

    return attributions

def merge_wordpiece_tokens(attributions):
    """
    Merges WordPiece tokens like '##ing' into the previous token.
    Input: list of (token, score)
    Output: list of (merged_token, aggregated_score)
    """
    merged = []
    current_token = ""
    current_score = 0.0

    for tok, score in attributions:
        if tok.startswith("##"):
            # continuation of previous word
            tok = tok[2:]
            current_token += tok
            current_score += score
        else:
            # push previous word
            if current_token:
                merged.append((current_token, current_score))
            current_token = tok
            current_score = score

    # push last
    if current_token:
        merged.append((current_token, current_score))

    return merged

def clean_special_tokens(attributions):
    return [(tok, score) for tok, score in attributions if tok not in ["[CLS]", "[SEP]"]]

def top_k_attributions(attributions, k=20):
    cleaned = []
    for tok, score in attributions:
        try:
            cleaned.append((tok, float(score)))
        except:
            continue  # skip tokens with invalid scores
    return sorted(cleaned, key=lambda x: abs(x[1]), reverse=True)[:k]



