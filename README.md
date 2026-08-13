# TextGuardAI
# SMS Spam Classification — NLP Pipeline + Flask Deployment

A full pipeline that takes raw SMS text through preprocessing, feature extraction, five
different modeling approaches (Naive Bayes, Linear SVM, GloVe embeddings, LSTM,
fine-tuned DistilBERT), and a Flask web app that serves the best practical model.

---

## 1. Project Structure

```
TextGuardAI
├── data/
│   ├── raw/                     # spam.csv, untouched
│   └── processed/               # cleaned dataset + saved feature matrices
├── src/
│   ├── preprocessing.py         # clean_text(), tokenization, lemmatization
│   ├── features.py              # TF-IDF / BoW vectorizer builders
│   ├── modeling.py              # train_and_evaluate() for classical approaches
│   ├── lstm_model.py            # tokenizer, padding, LSTM architecture, training
│   └── train_and_save.py        # trains final pipeline, saves to models/
├── models/
│   └── spam_pipeline.joblib     # TF-IDF vectorizer + Linear SVC, one object
├── results/
│   └── model_comparison.csv     # all approaches x all metrics
├── templates/
│   └── index.html               # Flask front-end
├── reports/
│   └── final_report.pdf         # full write-up (see also final_report.pdf in this repo)
├── app.py                       # Flask application entry point
├── main.py                      # training pipeline entry point
├── pyproject.toml
├── requirements.txt             # exported for deployment platforms
└── README.md
```

---

## 2. Environment Setup

```bash
# from the project root
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

Dependencies are pinned in `pyproject.toml` / `requirements.txt`. Core libraries: `pandas`,
`scikit-learn`, `nltk`, `gensim` (or raw GloVe file loading), `tensorflow`/`keras`,
`transformers`, `datasets`, `torch`, `joblib`, `flask`.

On first run, NLTK data is downloaded automatically (`punkt`, `stopwords`, `wordnet`,
`omw-1.4`) into your local NLTK data directory — no manual step needed.

### Dataset

Download `spam.csv` from Kaggle and place it at `data/raw/spam.csv`:

```bash
kaggle datasets download -d uciml/sms-spam-collection-dataset
```

Or manually from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

> **Note:** the raw file uses `encoding='latin-1'` — reading it with the pandas default
> (`utf-8`) will throw a `UnicodeDecodeError`.

### GloVe embeddings (Step 5 only)

Download `glove.6B.100d.txt` (part of the [GloVe 6B](https://nlp.stanford.edu/projects/glove/)
release) and place it at `embeddings/glove.6B.100d.txt` before running
`src/embeddings.py`.

---

## 3. Training the Models

Run the pipeline stages in order:

```bash
python src/preprocessing.py     # clean the raw dataset -> data/processed/
python src/features.py          # build & save TF-IDF / BoW vectorizers + matrices
python src/train_and_save.py    # train Naive Bayes + Linear SVC, save models/*.joblib
python src/embeddings.py        # GloVe averaging + Naive Bayes comparison
python src/lstm_model.py        # train & save the LSTM classifier
```

Or, to run everything end-to-end (preprocessing → features → classical models):

```bash
python main.py
```

The DistilBERT fine-tuning step is run separately in
Google Colab (GPU recommended) — see Section 5 below.

Each script prints its own accuracy / classification report and saves its artifacts under
`models/` and `results/` so later stages don't need to recompute earlier ones.

---

## 4. Running the Flask App

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser, paste or type an SMS message into the
text box, and submit. The app returns **Spam** or **Ham** plus a confidence score.

The app loads `models/spam_pipeline.joblib` — a single `scikit-learn Pipeline` containing
the fitted TF-IDF vectorizer *and* the trained Linear SVC together, so the exact same
preprocessing used during training is guaranteed at inference time. `app.py` imports
`clean_text()` directly from `src/preprocessing.py` rather than redefining it, so
training-time and serving-time cleaning can never drift apart.

**Quick test cases to try:**
- A normal short message (e.g. `"call you later?"`)
- A classic promo-style spam (e.g. `"WINNER!! You have been selected to receive a £900 prize reward, call now!"`)
- An edge case with lots of digits/links, and a very short message — these are the cases most
  likely to expose a preprocessing bug (see the misclassified examples in the final report).

---

## 5. Final Model Comparison

| Model                            | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|-----------------------------------|:--------:|:---------------:|:-----------:|:-------:|
| TF-IDF + Multinomial Naive Bayes  | 0.9650   | 0.99            | 0.74        | 0.85    |
| **TF-IDF + Linear SVC**           | **0.9883** | **0.986**      | **0.926**   | **0.955** |
| GloVe (avg.) + Multinomial NB     | 0.8664   | 0.00            | 0.00        | 0.00    |
| LSTM                               | 0.9830   | 0.96            | 0.91        | 0.93    |
| DistilBERT (fine-tuned)           | 0.9901   | 0.9792          | 0.9463      | 0.9625  |

Full breakdown, training curves, misclassified-example analysis, and answers to the six
required analysis questions are in **[final_report.pdf](final_report.pdf)**.

### Recommendation

**TF-IDF + Linear SVC is the model served in production (`app.py`).** It reaches 0.955
spam F1 — within ~0.8 points of fine-tuned DistilBERT's 0.963 — while being far cheaper to
run: inference is a single sparse matrix-vector multiply (sub-millisecond on CPU, no GPU
needed), and the whole pipeline serializes to a few megabytes with no model download or
warm-up cost. DistilBERT's small accuracy edge doesn't clearly justify its GPU-oriented
inference cost and much larger artifact for this dataset/use case. LSTM underperforms both
and shows clear overfitting on a dataset this size (see the training curve in the report),
so it isn't recommended for either accuracy or cost reasons.

---

## 6. Known Gaps / Next Steps

- **BoW vs TF-IDF model comparison**: Bag-of-Words feature matrices are built and saved, but
  a classifier was not yet trained/scored on them — train the same Naive Bayes / Linear SVC
  on `X_train_bow` / `X_test_bow` (already in `models/`) to complete this comparison.
- **Top-15 spam-indicative words**: not yet extracted from the trained models' coefficients —
  sort `LinearSVC.coef_` (or the NB `feature_log_prob_` difference between classes) against
  `TfidfVectorizer.get_feature_names_out()` to generate this list.

---

## 7. GitHub

Push the repository and share the link once complete. `.gitignore` excludes `data/`,
`models/*.joblib`, and other large generated artifacts from version control.