import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Simple dataset
data = {
    "text": [
        "I feel happy today",
        "Life is great",
        "I am relaxed",
        "I am stressed",
        "Work pressure is too much",
        "I feel overwhelmed",
        "I am sad",
        "I am excited",
        "Everything is amazing",
        "I feel anxious"
    ],
    "label":[0,0,0,1,1,1,1,0,0,1]
}

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model
pickle.dump(model, open("model.pkl","wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("Model and vectorizer saved successfully")