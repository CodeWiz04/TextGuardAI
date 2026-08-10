import joblib
import numpy as np

# Load train/test data from Step 4
X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

def load_glove(path):
    embeddings={}r
    with open(path,'r',encoding='utf-8') as f:  #encoding='utf-8' tells how the bytes in the file should be interpreted as characters 
        for line in f:
            values=line.split() 
            word=values[0]                        #extract word
            vector=np.asarray(values[1:],'float32')#extract vector and convert to numpy array of type float32
            embeddings[word]=vector                #insert a dictionary entry with the word as key and the vector as value


glove = load_glove("embeddings/glove.6B.100d.txt")

joblib.dump(glove, "models/glove_embeddings.pkl")

print("GloVe embeddings saved.")
print("Vocabulary size:", len(glove))


