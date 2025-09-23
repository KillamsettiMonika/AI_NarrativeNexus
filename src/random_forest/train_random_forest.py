import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Path to dataset
dataset_path = os.path.join("src", "random_forest", "two_column_dataset.csv")

# Path to save model
model_path = os.path.join("models", "random_forest_model.pkl")

def train_random_forest():
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Extract features and labels
    X = df["text"].astype(str)   # Ensure it's string
    y = df["category"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_vec, y_train)

    # Predictions
    y_pred = clf.predict(X_test_vec)

    # Evaluation
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))

    # Save model + vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, os.path.join("models", "rf_vectorizer.pkl"))
    print(f"\n💾 Model saved to {model_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues")
    plt.title("Random Forest Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("src/random_forest/confusion_matrix_rf.png")
    plt.show()


if __name__ == "__main__":
    train_random_forest()
