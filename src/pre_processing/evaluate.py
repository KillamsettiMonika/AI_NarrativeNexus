import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load dataset (use cleaned data)
DATASET_PATH = "cleaned_dataset.csv"
MODEL_PATH = "models/text_classifier.pkl"

print(f"📂 Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# Drop rows with missing values
df = df.dropna(subset=["clean_text", "category"])

X = df["clean_text"]
y = df["category"]

# Train/test split (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Load trained model
print(f"💾 Loading model: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# 3. Evaluate model
print("📊 Evaluating model...")
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 4. Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
