import joblib
import os

# Paths to model & vectorizer
model_path = os.path.join("models", "random_forest_model.pkl")
vectorizer_path = os.path.join("models", "rf_vectorizer.pkl")

# Load model & vectorizer
clf = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_category(text: str):
    """Predict newsgroup category for a given text"""
    X_vec = vectorizer.transform([text])
    pred = clf.predict(X_vec)[0]
    return pred

if __name__ == "__main__":
    # Try some test predictions
    samples = [
        "The new graphics card has amazing rendering performance.",
        "The church service discussed forgiveness and faith.",
        "NASA announced a new Mars rover mission.",
        "Gun control policies are being debated in congress."
    ]

    for text in samples:
        print(f"\n📝 Text: {text}")
        print(f"👉 Predicted Category: {predict_category(text)}")

