import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths
test_path = "datasets/amazon_rev/amazon_reviews_test.csv"
model_path = "models/amazon_lstm.h5"
tokenizer_path = "models/amazon_lstm_tokenizer.pkl"
cm_path = "src/lstm/confusion_matrix_lstm.png"

# Params
MAX_WORDS = 20000
MAX_LEN = 200

# Load model + tokenizer
print("📂 Loading model and tokenizer...")
model = tf.keras.models.load_model(model_path)
tokenizer = joblib.load(tokenizer_path)

# Load test data
print("📂 Loading test data...")
test_df = pd.read_csv(test_path, nrows=10000)
test_df["text"] = (test_df["title"].astype(str) + " " + test_df["content"].astype(str))

X_test, y_test = test_df["text"], test_df["label"]

# Tokenize & pad
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

# Predictions
print("🔎 Evaluating LSTM...")
y_pred_probs = model.predict(X_test_seq)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Metrics
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

os.makedirs(os.path.dirname(cm_path), exist_ok=True)
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"💾 Confusion matrix saved at {cm_path}")
