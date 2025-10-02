import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Paths
DATASET_PATH = "cleaned_dataset.csv"   # path relative to project root
MODEL_PATH = "models/text_classifier.pkl"

print(f"📂 Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# ---------------------------
# Auto-detect text column
# ---------------------------
text_column = None
for col in ["clean_text", "text", "content", "body"]:
    if col in df.columns:
        text_column = col
        break

if not text_column:
    raise ValueError(
        f"❌ No valid text column found! Available columns: {df.columns.tolist()}"
    )

print(f"✅ Using text column: {text_column}")

# Drop missing rows
df = df.dropna(subset=[text_column, "category"])

X = df[text_column]
y = df["category"]

# Split (same as in training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load model
print(f"💾 Loading model: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Predictions
print("📊 Evaluating model...")
y_pred = model.predict(X_test)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# ✅ Save confusion matrix to file
plt.savefig("models/confusion_matrix.png", dpi=300)
plt.show()

print("✅ Confusion matrix saved to models/confusion_matrix.png")
