import os
import pandas as pd
from transformers import pipeline

# ---------- CONFIG ----------
DATASET_PATH = "datasets/BBC articles/bbc-text.csv"  # adjust if path differs
OUTPUT_PATH = "text_summarization/abstractive_summary.csv"

# ---------- LOAD DATA ----------
print("📥 Loading dataset...")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
text_column = None

# Automatically detect text column
for col in df.columns:
    if df[col].dtype == 'object' and df[col].str.len().mean() > 20:
        text_column = col
        break

if not text_column:
    raise ValueError("No suitable text column found in the dataset!")

print(f"✅ Using text column: '{text_column}'")

# ---------- LOAD MODEL ----------
print("🔄 Loading summarization model (facebook/bart-large-cnn)...")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    framework="pt"  # force PyTorch backend (no TensorFlow issues)
)
print("✅ Model loaded successfully!")

# ---------- GENERATE SUMMARIES ----------
summaries = []
print("⚙️ Generating summaries... (this might take a few minutes for large data)")
for i, text in enumerate(df[text_column].astype(str).tolist()[:10]):  # summarize first 10 for demo
    try:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
        print(f"📝 {i+1}. Summary generated.")
    except Exception as e:
        summaries.append(f"[Error summarizing text {i+1}]: {e}")

# ---------- SAVE OUTPUT ----------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for i, summary in enumerate(summaries, 1):
        f.write(f"{i}. {summary}\n\n")

print(f"✅ Summaries saved to: {OUTPUT_PATH}")
