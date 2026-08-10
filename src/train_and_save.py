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

nb_model=MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions=nb_model.predict(X_test_tfidf)

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

svm_model=LinearSVC(class_weight="balanced",random_state=42)
svm_model.fit(X_train_tfidf, y_train)
svm_predictions=svm_model.predict(X_test_tfidf)
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