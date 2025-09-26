import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths
test_path = "datasets/amazon_rev/amazon_reviews_test.csv"
model_path = "models/amazon_rf_model.pkl"
vectorizer_path = "models/amazon_tfidf.pkl"
cm_path = "src/random_forest1/confusion_matrix_rf.png"   # Save confusion matrix here

# Load model + vectorizer
print("📂 Loading model and vectorizer...")
rf = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Load dataset
print("📂 Loading test data...")
test_df = pd.read_csv(test_path, nrows=10000)
test_df["text"] = (test_df["title"].astype(str) + " " + test_df["content"].astype(str))

X_test, y_test = test_df["text"], test_df["label"]

# Transform test set
X_test_tfidf = vectorizer.transform(X_test)

# Predictions
print("🔎 Evaluating Random Forest...")
y_pred = rf.predict(X_test_tfidf)

# Metrics
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot with seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# Save instead of showing
os.makedirs(os.path.dirname(cm_path), exist_ok=True)
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"💾 Confusion matrix saved at {cm_path}")
