"""
Flask app that serves the fine-tuned DistilBERT spam classifier.
Loads the model + tokenizer saved in Step 7 (models/distilbert_spam/)
and reuses the exact same clean_text() preprocessing used at training time.
"""

from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocessing import clean_text  # same function used in training

MODEL_DIR = "models/distilbert_spam"
MAX_LEN = 128  # must match the max_length used when training (Step 7, tokenize_function)

app = Flask(__name__)

# Load once at startup, not per-request -- loading a transformer model is
# expensive (hundreds of MB), so we keep it in memory across requests.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()  # inference mode: disables dropout, etc.
except Exception as e:
    raise RuntimeError(
        f"Could not load model from {MODEL_DIR}. "
        f"Make sure Step 7 has been run and the model was saved there. Error: {e}"
    )

LABELS = {0: "Ham", 1: "Spam"}


def predict_message(raw_text: str):
    """
    Runs the full inference pipeline on a single raw message:
    clean -> tokenize -> model forward pass -> softmax -> label + confidence.
    Returns (label_str, confidence_float) or raises ValueError on empty input.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Message is empty.")

    # Same cleaning function used to build clean_message during training --
    # this is the step most people forget, and it silently wrecks accuracy.
    cleaned = clean_text(raw_text)

    # Tokenize exactly like training: same max_length, padding, truncation
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

    with torch.no_grad():  # no gradient tracking needed at inference, saves memory/time
        outputs = model(**inputs)   #inputs = {"input_ids": ...,"attention_mask": ...} The ** simply unpacks the dictionary into keyword arguments
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze() #for example if model returns [1,2] shape it converts into [2] which is easier to work with:[Ham probability, Spam probability]

    pred_class = int(torch.argmax(probs).item()) #pytorch returns a tensor so .item converts into normal int
    confidence = float(probs[pred_class].item()) #returns probability of predicted class

    return LABELS[pred_class], confidence


@app.route("/", methods=["GET", "POST"])
def index():
    result_label = None
    confidence_pct = None
    error = None
    submitted_text = ""

    if request.method == "POST":
        submitted_text = request.form.get("message", "")
        try:
            result_label, confidence = predict_message(submitted_text)
            confidence_pct = round(confidence * 100, 2)
        except ValueError as e:
            error = str(e)
        except Exception as e:
            # Catch-all so a bad request never shows the user a raw stack trace
            error = f"Something went wrong during prediction: {e}"

    return render_template(
        "index.html",
        result_label=result_label,
        confidence_pct=confidence_pct,
        error=error,
        submitted_text=submitted_text
    )


if __name__ == "__main__":
    app.run(debug=True)