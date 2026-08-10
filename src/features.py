import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

df = pd.read_csv("data/processed/cleaned_spam.csv")
print(df["clean_message"].isna().sum())
#Split the data
X=df['clean_message']
y=df['label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

tfidf=TfidfVectorizer(max_features=5000)     #Collects unique words, find count occurences of each word,form it into vector called bag of words
                                             #but bag of words isn't enough as all the words get same importance, so we use tfidf which gives weightage to words based on their importance
                                             #Tf=>how important is this word inside the document(Number of times word appears/Total words in document)
                                             #idf=>gives more importance to rare words(log(Total Documents/Number of documents containing the word))
                                             #max_features=>only consider top 5000 words based on their importance   reduce memory usage,speedup


X_train_tfidf=tfidf.fit_transform(X_train)   #learns the vocab,compute idf scores,converts message into tfidf vector
X_test_tfidf=tfidf.transform(X_test)         #converts message into tfidf vector using the vocab learned from training data  

bow=CountVectorizer(max_features=5000)       #bag of words model
X_train_bow=bow.fit_transform(X_train)
X_test_bow=bow.transform(X_test)

print("TF-IDF Train:", X_train_tfidf.shape)
print("TF-IDF Test :", X_test_tfidf.shape)

print("BoW Train:", X_train_bow.shape)
print("BoW Test :", X_test_bow.shape)

os.makedirs("models", exist_ok=True)
# Save TF-IDF features
joblib.dump(X_train_tfidf, "data/processed/X_train_tfidf.pkl")
joblib.dump(X_test_tfidf, "data/processed/X_test_tfidf.pkl")

# Save BoW features
joblib.dump(X_train_bow, "data/processed/X_train_bow.pkl")
joblib.dump(X_test_bow, "data/processed/X_test_bow.pkl")

# Save labels
joblib.dump(y_train, "data/processed/y_train.pkl")
joblib.dump(y_test, "data/processed/y_test.pkl")

joblib.dump(X_train_tfidf, "models/X_train_tfidf.pkl")
joblib.dump(X_test_tfidf, "models/X_test_tfidf.pkl")

joblib.dump(X_train_bow, "models/X_train_bow.pkl")
joblib.dump(X_test_bow, "models/X_test_bow.pkl")

joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

# Save vectorizers
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(bow, "models/bow_vectorizer.pkl")

print("\nAll features and vectorizers saved successfully.")