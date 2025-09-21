import pandas as pd
import re


def clean_body_classification(raw_text: str) -> str:
    """
    Clean raw text for classification.
    Removes headers, quotes, signatures, metadata, and normalizes text.
    """
    if not raw_text or pd.isna(raw_text):
        return ""

    # Drop quoted lines and reply boilerplate
    cleaned_lines = []
    for line in raw_text.splitlines():
        if line.strip().startswith((">", "|")):
            continue
        if re.search(r"(writes:|wrote:|In article\s*<.*?>)", line, re.I):
            continue
        cleaned_lines.append(line)

    body = "\n".join(cleaned_lines)

    # Remove emails, urls, html tags, numbers, and special chars
    body = re.sub(r"\b\S+@\S+\b", " ", body)
    body = re.sub(r"http\S+|www\.\S+", " ", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\d+", " ", body)
    body = re.sub(r"[^a-zA-Z\s]", " ", body)

    # Normalize spacing
    body = re.sub(r"\s{2,}", " ", body)

    return body.lower().strip()


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    input_file = "20news_18828_final_2000.csv"
    output_file = "cleaned_dataset.csv"

    print(f"📂 Loading dataset: {input_file}")
    df = pd.read_csv(input_file)

    # Expect columns: ["filename", "category", "text"] or ["category", "text"]
    if "text" not in df.columns:
        raise ValueError("❌ Input CSV must have a 'text' column!")

    print("✨ Cleaning text...")
    df["text"] = df["text"].map(clean_body_classification)

    # Drop empty rows
    df = df[df["text"].str.strip().astype(bool)]

    # Drop categories with <2 samples
    df = df.groupby("category").filter(lambda x: len(x) > 1)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Saved cleaned dataset: {output_file}")
    print(f"   Rows: {len(df)}, Categories: {df['category'].nunique()}")
