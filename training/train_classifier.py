# training/train_classifier.py

import os
import pandas as pd
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Ensure NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

INPUT_CSV = "cleaned_dataset.csv"
MODEL_PATH = os.path.join("models", "text_classifier.pkl")
CONF_MATRIX_PATH = os.path.join("models", "confusion_matrix.png")

# ===================
# 1. Load Dataset
# ===================
print(f"📂 Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# 🚨 FIX: Drop rows with missing or empty text
df = df.dropna(subset=["clean_text"])
df = df[df["clean_text"].str.strip() != ""]

# Drop categories with <2 samples
df = df.groupby("category").filter(lambda x: len(x) > 1)

X = df["clean_text"]
y = df["category"]

print(f"📊 Dataset after cleaning: {len(df)} rows, {y.nunique()} categories")

# ===================
# 2. Train-Test Split
# ===================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================
# 3. Pipeline
# ===================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

print("Training pipeline...")
pipeline.fit(X_train, y_train)

# ===================
# 4. Evaluation
# ===================
y_pred = pipeline.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.close()

print(f"✅ Confusion matrix saved to {CONF_MATRIX_PATH}")

# ===================
# 5. Save Model
# ===================
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
