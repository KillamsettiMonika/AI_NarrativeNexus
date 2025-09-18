import os
import pandas as pd
import csv
from pre_processing.text_processing import nlp_preprocess

# Input & Output paths
INPUT_CSV = os.path.join("20news_18828_final_2000.csv")
OUTPUT_CSV = os.path.join("cleaned_dataset.csv")

def clean_dataset(input_csv, output_csv):
    print(f"📂 Loading dataset: {os.path.abspath(input_csv)}")
    df = pd.read_csv(input_csv)

    print("🧹 Applying text preprocessing...")
    df["clean_text"] = df["text"].astype(str).apply(nlp_preprocess)

    # 🔑 Fix: Replace line breaks inside text with a space
    df["clean_text"] = df["clean_text"].str.replace(r"\r?\n", " ", regex=True)
    df["text"] = df["text"].astype(str).str.replace(r"\r?\n", " ", regex=True)

    # Save with quotes around all fields (Excel-safe)
    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL
    )

    print(f"✅ Cleaned dataset saved: {os.path.abspath(output_csv)}")
    print(df.head())

if __name__ == "__main__":
    clean_dataset(INPUT_CSV, OUTPUT_CSV)
