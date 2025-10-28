# text_summarization/extractive_summarizer_bbc.py
import os
import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk

nltk.download("punkt", quiet=True)

# === Paths ===
DATASET_PATH = "datasets/BBC articles/bbc-text.csv"  # Change this if needed
OUTPUT_PATH = "text_summarization/bbc_extractive_summary.csv"

# === Function ===
def extractive_summary(text, sentence_count=3):
    """Generate extractive summary using LexRank"""
    parser = PlaintextParser.from_string(str(text), Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count=sentence_count)
    return " ".join(str(sentence) for sentence in summary)

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    if "text" not in df.columns:
        print("❌ Expected a column named 'text' in your dataset.")
        return

    print(f"📊 Loaded {len(df)} BBC articles.")
    df["extractive_summary"] = df["text"].apply(lambda x: extractive_summary(x))

    os.makedirs("text_summarization", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Extractive summaries saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
