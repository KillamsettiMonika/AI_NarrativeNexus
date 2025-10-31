import os
import pandas as pd
import joblib
import numpy as np
import evaluate
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ---------- CONFIG ----------
DATASET_PATH = "datasets/BBC articles/bbc-text.csv"  # adjust if path differs
OUTPUT_PATH = "text_summarization/abstractive_summary.csv"
MODEL_PATH = "models/abstractive_model.pkl"
CONFUSION_MATRIX_PATH = "models/confusion_matrix.png"

# ---------- LOAD DATA ----------
print("📥 Loading dataset...")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

# Automatically detect the main text column
text_column = None
for col in df.columns:
    if df[col].dtype == "object" and df[col].str.len().mean() > 20:
        text_column = col
        break

if not text_column:
    raise ValueError("❌ No suitable text column found in dataset!")

print(f"✅ Using text column: '{text_column}'")

# ---------- LOAD OR CREATE MODEL ----------
os.makedirs("models", exist_ok=True)

if os.path.exists(MODEL_PATH):
    print("📦 Loading summarization model from local file...")
    summarizer = joblib.load(MODEL_PATH)
else:
    print("🔄 Downloading summarization model (facebook/bart-large-cnn)...")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        framework="pt"
    )
    joblib.dump(summarizer, MODEL_PATH)
    print(f"✅ Model saved locally to: {MODEL_PATH}")

# ---------- GENERATE SUMMARIES ----------
summaries = []
print("⚙️ Generating summaries... (processing first 10 for demo)")
for i, text in enumerate(df[text_column].astype(str).tolist()[:10]):
    try:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
        print(f"📝 {i + 1}. Summary generated.")
    except Exception as e:
        summaries.append(f"[Error summarizing text {i + 1}]: {e}")

# ---------- SAVE OUTPUT ----------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

summary_df = pd.DataFrame({
    "original_text": df[text_column].astype(str).tolist()[:10],
    "summary": summaries
})
summary_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"✅ Summaries saved to: {OUTPUT_PATH}")

# ---------- EVALUATE MODEL WITH ROUGE ----------
print("\n📊 Evaluating summarization quality using ROUGE...")
rouge = evaluate.load("rouge")

references = [text.split('.')[0] for text in df[text_column].astype(str).tolist()[:10]]
results = rouge.compute(predictions=summaries, references=references)

rouge_scores = {k: np.mean(v) for k, v in results.items()}

print("\n📈 ROUGE Evaluation Results:")
for metric, score in rouge_scores.items():
    print(f"  {metric.upper()}: {score:.4f}")

# ---------- CONFUSION MATRIX (Cluster Comparison) ----------
print("\n🧩 Creating confusion matrix based on text similarity clusters...")

vectorizer = TfidfVectorizer(stop_words='english')
X_orig = vectorizer.fit_transform(df[text_column].astype(str).tolist()[:10])
X_sum = vectorizer.transform(summaries)

# cluster original and summarized text
kmeans_orig = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_sum = KMeans(n_clusters=3, random_state=42, n_init=10)
orig_labels = kmeans_orig.fit_predict(X_orig)
sum_labels = kmeans_sum.fit_predict(X_sum)

cm = confusion_matrix(orig_labels, sum_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix — Summary vs Original Clusters")
plt.savefig(CONFUSION_MATRIX_PATH)
plt.close()

print(f"✅ Confusion Matrix saved at: {CONFUSION_MATRIX_PATH}")
print("🎯 All tasks completed successfully!")
