import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score,precision_score, recall_score, f1_score



# --------------------------------------------------
# 1. Load the original cleaned text dataset
# --------------------------------------------------

df = pd.read_csv("data/processed/cleaned_spam.csv")

X = df["clean_message"]
y = df["label"]


# --------------------------------------------------
# 2. Create the SAME train/test split as Step 4
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Load GloVe embeddings
# --------------------------------------------------

def load_glove(path):

    embeddings = {}

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            values = line.split()

            word = values[0]

            vector = np.asarray(
                values[1:],
                dtype="float32"
            )

            embeddings[word] = vector

    return embeddings


glove = load_glove("embeddings/glove.6B.100d.txt")

print("GloVe vocabulary size:", len(glove))

# --------------------------------------------------
# 4. Convert one message into one GloVe vector
# ----------------------------------------------
def message_to_vector(message,glove,vector_size=100):
    words=message.split()
    vectors=[]
    for word in words:
        if word in glove:
            vectors.append(glove[word])
        if len(vectors)==0:
            return np.zeros(vector_size)
    return np.mean(vectors,axis=0)
# --------------------------------------------------
# 5. Convert all messages into GloVe vectors
# --------------------------------------------------

X_train_glove = np.array([
    message_to_vector(message, glove)
    for message in X_train
])

X_test_glove = np.array([
    message_to_vector(message, glove)
    for message in X_test
])


print("X_train GloVe shape:", X_train_glove.shape)
print("X_test GloVe shape:", X_test_glove.shape)

# --------------------------------------------------
# 6. Make GloVe values non-negative
# --------------------------------------------------

scaler = MinMaxScaler()    #shifts the values in the range of [0,1]   x`=(x-xmin)/(xmax-xmin)

X_train_glove = scaler.fit_transform(X_train_glove)  #finds max and min of all 100-dim vectors

X_test_glove = scaler.transform(X_test_glove)        #Uses those values to convert the training vectors into the [0, 1] range.


# --------------------------------------------------
# 7. Train Multinomial Naive Bayes
# --------------------------------------------------

nb_model = MultinomialNB()

nb_model.fit(
    X_train_glove,
    y_train
)

# --------------------------------------------------
# 8. Predict test messages
# --------------------------------------------------

glove_predictions = nb_model.predict(X_test_glove)

# --------------------------------------------------
# 9. Evaluate
# --------------------------------------------------

print("\n" + "=" * 60)
print("GLOVE + MULTINOMIAL NAIVE BAYES")
print("=" * 60)

print(
    f"Accuracy: "
    f"{accuracy_score(y_test, glove_predictions):.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        glove_predictions
    )
)

joblib.dump(
    glove,
    "models/glove_embeddings.pkl"
)

joblib.dump(
    scaler,
    "models/glove_scaler.pkl"
)

joblib.dump(
    nb_model,
    "models/glove_naive_bayes.pkl"
)

joblib.dump(
    X_train_glove,
    "models/X_train_glove.pkl"
)

joblib.dump(
    X_test_glove,
    "models/X_test_glove.pkl"
)
# Load Step 4 artifacts
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_model = joblib.load("models/naive_bayes.pkl")

X_test_tfidf = joblib.load("models/X_test_tfidf.pkl")
y_test = joblib.load("models/y_test.pkl")

# Step 4 predictions
tfidf_predictions = tfidf_model.predict(X_test_tfidf)

# Step 5 predictions
glove_model = joblib.load("models/glove_naive_bayes.pkl")

X_test_glove = joblib.load("models/X_test_glove.pkl")

glove_predictions = glove_model.predict(X_test_glove)

# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "TF-IDF + MultinomialNB",
        "GloVe + MultinomialNB"
    ],

    "Accuracy": [
        accuracy_score(y_test, tfidf_predictions),
        accuracy_score(y_test, glove_predictions)
    ],

    "Spam Precision": [
        precision_score(
            y_test,
            tfidf_predictions,
            pos_label="spam",
            zero_division=0
        ),
        precision_score(
            y_test,
            glove_predictions,
            pos_label="spam",
            zero_division=0
        )
    ],

    "Spam Recall": [
        recall_score(
            y_test,
            tfidf_predictions,
            pos_label="spam",
            zero_division=0
        ),
        recall_score(
            y_test,
            glove_predictions,
            pos_label="spam",
            zero_division=0
        )
    ],

    "Spam F1": [
        f1_score(
            y_test,
            tfidf_predictions,
            pos_label="spam",    #taking spam as positive class(out of all the predicted spams,how many were actually spam)
            zero_division=0      #return 0 if 0/0 situation occurs (i.e. if there are no predicted spams)
        ),
        f1_score(
            y_test,
            glove_predictions,
            pos_label="spam",
            zero_division=0
        )
    ]
})

print("\n" + "=" * 70)
print("TF-IDF vs GloVe COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)
# --------------------------------------------------
# Save comparison
# --------------------------------------------------

comparison.to_csv(
    "results/tfidf_vs_glove_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("results/tfidf_vs_glove_comparison.csv")
print("\nStep 5 models and embeddings saved successfully.")