import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


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
