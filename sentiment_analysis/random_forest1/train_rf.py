import os
import re
import nltk
import pandas as pd
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Download stopwords/lemmatizer resources if not already
nltk.download("stopwords")
nltk.download("wordnet")

# Paths
train_path = "datasets/amazon_rev/amazon_reviews_train.csv"
test_path  = "datasets/amazon_rev/amazon_reviews_test.csv"
model_path = "models/amazon_rf_model.pkl"
vectorizer_path = "models/amazon_tfidf.pkl"

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# Load dataset
print("📂 Loading training data...")
train_df = pd.read_csv(train_path, nrows=50000)
test_df  = pd.read_csv(test_path, nrows=10000)

train_df["text"] = (train_df["title"].astype(str) + " " + train_df["content"].astype(str)).apply(clean_text)
test_df["text"]  = (test_df["title"].astype(str) + " " + test_df["content"].astype(str)).apply(clean_text)

X_train, y_train = train_df["text"], train_df["label"]

# Vectorize
print("🔧 Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train model
print("🚀 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_tfidf, y_train)

# Save model + vectorizer
joblib.dump(rf, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"✅ Model saved at {model_path}")
print(f"✅ Vectorizer saved at {vectorizer_path}")
