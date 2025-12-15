# # gating/train_gating.py
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import joblib

# # Load dataset
# df = pd.read_csv("/home/dtp2025/Documents/moe/dataset.csv")
# X = df["query"]
# y = df["expert"]

# # Vectorize text
# vectorizer = TfidfVectorizer()
# X_vec = vectorizer.fit_transform(X)

# # Train classifier
# clf = MultinomialNB()
# clf.fit(X_vec, y)

# # Save model and vectorizer
# joblib.dump(clf, "../configs/gating_model.joblib")
# joblib.dump(vectorizer, "../configs/vectorizer.joblib")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ======== Load Dataset ========
df = pd.read_csv("dataset.csv")

X = df["query"]
y = df["expert"]

# ======== Split Data ========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======== Vectorization ========
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ======== Train Model ========
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# ======== Evaluate Model ========
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ======== Save Model ========
os.makedirs("configs", exist_ok=True)
joblib.dump(clf, "configs/gating_model.joblib")
joblib.dump(vectorizer, "configs/vectorizer.joblib")

print("✅ Gating model and vectorizer saved successfully!")
