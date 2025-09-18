import os
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Ensure nltk data is available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

from pre_processing.text_processing import nlp_preprocess

# Paths
INPUT_CSV = "cleaned_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "text_classifier.pkl")


def build_pipeline():
    """Builds the ML pipeline with TF-IDF + Logistic Regression."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    return pipeline


def train_and_save(X, y):
    """Train classifier and save model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    print(f"📂 Loading dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Ensure required columns
    if "clean_text" not in df.columns or "category" not in df.columns:
        raise ValueError("CSV must have columns: clean_text, category")

    # ⚠️ Drop rows with missing/empty text or category
    df = df.dropna(subset=["clean_text", "category"])
    df = df[df["clean_text"].str.strip() != ""]

    # ⚠️ Remove categories with <2 samples (required for stratify)
    df = df.groupby("category").filter(lambda x: len(x) > 1)

    print(f"📊 Dataset after cleaning: {len(df)} rows, {df['category'].nunique()} categories")

    X = df["clean_text"]
    y = df["category"]

    train_and_save(X, y)
