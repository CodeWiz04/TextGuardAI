import joblib
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, classification_report

X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

y_train = (y_train == "spam").astype(int)
y_test = (y_test == "spam").astype(int)

# --------------------------------------------------
# Tokenization
# --------------------------------------------------
tokenizer=Tokenizer(
    num_words=10000,               #keep 10000 most frequent words
    oov_token="<OOV>"              #label for out-of-vocabulary words(typically 1 is reserved for unknown words,0 for padding)
)

tokenizer.fit_on_texts(X_train)    #count the words in the training set and create a word index
X_train_sequences=tokenizer.texts_to_sequences(X_train) #looks for each word in the dict and gives that particular index
X_test_sequences=tokenizer.texts_to_sequences(X_test)   #use the knowledge learn from training sets and form vectors

# --------------------------------------------------
# Padding
# --------------------------------------------------
MAX_LEN=50     #each vector can have upto 50 words, if less than 50 words then pad with 0s, if more than 50 words then truncate the extra words

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LEN,
    padding="post",          #pad 0s at the end of the vector
    truncating="post"        #truncate the extra words at the end of the vector
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

print("X_train padded shape:", X_train_padded.shape)
print("X_test padded shape:", X_test_padded.shape)


# --------------------------------------------------
# Build LSTM model
# --------------------------------------------------

vocab_size=min(10000, len(tokenizer.word_index)+1)  #how many different integer token IDs it needs to accommodate, while limiting the vocabulary to 10,000.

model=Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=100
    ),
    Bidirectional(
        LSTM(64)
    ),
    Dropout(0.4),

    Dense(1,activation="sigmoid")
    
])