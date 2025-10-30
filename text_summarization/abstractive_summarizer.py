import os
import pandas as pd
import joblib
from transformers import pipeline

# ---------- CONFIG ----------
DATASET_PATH = "datasets/BBC articles/bbc-text.csv"  # adjust if path differs
OUTPUT_PATH = "text_summarization/abstractive_summary.csv"
MODEL_PATH = "models/abstractive_model.pkl"

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
        framework="pt"  # force PyTorch backend
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
print(f"✅ Model available in: {MODEL_PATH}")
