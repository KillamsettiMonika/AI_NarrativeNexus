import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from topicmodelling.pre_processing.text_processing import preprocess_series

# Paths
DATASET_PATH = "cleaned_dataset.csv"
MODEL_PATH = "models/text_classifier.pkl"

print(f"📂 Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# ---------------------------
# Handle missing column issue
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

# Remove categories with fewer than 2 samples
class_counts = df["category"].value_counts()
df = df[df["category"].isin(class_counts[class_counts >= 2].index)]

X = df[text_column]
y = df["category"]

print(f"📊 Dataset after cleaning: {len(df)} rows, {len(y.unique())} categories")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=200))
])

print("Training pipeline...")
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Accuracy: {pipeline.score(X_test, y_test):.4f}")
print(f"💾 Model saved to {MODEL_PATH}")
