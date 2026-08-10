import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ----------------------------
# Load saved artifacts
# ----------------------------
X_train_tfidf = joblib.load("models/X_train_tfidf.pkl")
X_test_tfidf = joblib.load("models/X_test_tfidf.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

os.makedirs("models", exist_ok=True)

nb_model=MultinomialNB()                      #probabilistic model for classification, uses bayes theorem in backend e.g. P(message|spam)=P(free|spam)*P(prize|winner)*P(winner|spam)
nb_model.fit(X_train_tfidf, y_train)          #it learns the prior prob probabilities of each class and the likelihood of each feature given the class
nb_predictions=nb_model.predict(X_test_tfidf) #Computes P(spam|message) and P(ham|message) and predicts the class with the higher probability

print("=" * 60)
print("MULTINOMIAL NAIVE BAYES")
print("=" * 60)

print(f"Accuracy: {accuracy_score(y_test, nb_predictions):.4f}\n")

print("Classification Report")
print(classification_report(y_test, nb_predictions))

print("Confusion Matrix")
print(confusion_matrix(y_test, nb_predictions))

joblib.dump(nb_model, "models/naive_bayes.pkl")

# ======================================================
# 2. Linear SVM
# ======================================================

svm_model=LinearSVC(class_weight="balanced",random_state=42)   #Creates a linear SVM classifier with balanced class weights to handle class imbalance and a fixed random state for reproducibility
svm_model.fit(X_train_tfidf, y_train)                          #tries to find the best weight and bias on which the margain between two categories is maximized. It learns the hyperplane that best separates the classes in the feature space
svm_predictions=svm_model.predict(X_test_tfidf)                #Classify the unseen messages 
print("\n")
print("=" * 60)
print("LINEAR SVC")
print("=" * 60)

print(f"Accuracy: {accuracy_score(y_test, svm_predictions):.4f}\n")

print("Classification Report")
print(classification_report(y_test, svm_predictions))

print("Confusion Matrix")
print(confusion_matrix(y_test, svm_predictions))

joblib.dump(svm_model, "models/linear_svm.pkl")

print("\nModels saved successfully!")