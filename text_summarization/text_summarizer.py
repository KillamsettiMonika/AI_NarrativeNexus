# text_summarization/bart_summarizer.py
import os
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration

# -----------------------------
# 1. Load model & tokenizer
# -----------------------------
print("Loading BART model and tokenizer...")
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# -----------------------------
# 2. Load your dataset
# -----------------------------
# Update path if needed
DATA_PATH = "datasets/BBC articles/bbc-text.csv"
 # or "datasets/CNN_data.csv"
df = pd.read_csv(DATA_PATH)

# Assuming your CSV has columns like 'text' or 'article' etc.
if "text" in df.columns:
    texts = df["text"].dropna().tolist()
elif "article" in df.columns:
    texts = df["article"].dropna().tolist()
elif "content" in df.columns:
    texts = df["content"].dropna().tolist()
else:
    raise ValueError("⚠️ Dataset must have a column like 'text', 'article', or 'content'.")

# -----------------------------
# 3. Summarization function
# -----------------------------
def summarize_text(text, max_len=130, min_len=30):
    """Generate summary for a given text using BART"""
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=max_len,
        min_length=min_len,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# -----------------------------
# 4. Generate summaries
# -----------------------------
summaries = []
for i, text in enumerate(texts[:20]):  # summarize first 20 for testing
    print(f"Summarizing article {i+1}/{len(texts)}...")
    summary = summarize_text(text)
    summaries.append(summary)

# -----------------------------
# 5. Save results
# -----------------------------
df_summary = pd.DataFrame({
    "original_text": texts[:20],
    "summary": summaries
})

OUTPUT_PATH = "datasets/summarized_output.csv"
df_summary.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n✅ Summaries saved to: {OUTPUT_PATH}")
print(df_summary.head())
