import os
import pandas as pd
import re

# Root dataset folder (update path if needed)
DATASET_DIR = os.path.join("datasets", "20news-18828")
OUTPUT_CSV = "20news_18828_final_2000.csv"

def clean_body(raw_text: str) -> str:
    """Deep clean the body text of a 20NG post."""
    if not raw_text:
        return ""

    # Split headers vs body (headers end at first blank line)
    parts = re.split(r"\n\s*\n", raw_text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]

    cleaned_lines = []
    for line in body.splitlines():
        # Skip metadata headers
        if re.match(
            r"^(archive-name|from|subject|path|xref|organization|lines|newsgroups|"
            r"message-id|keywords|last-modified|version):",
            line,
            re.I,
        ):
            continue
        # Skip quotes / replies
        if line.strip().startswith((">", "|")):
            continue
        # Skip signatures
        if line.strip().startswith("--"):
            break
        # Skip common "writes:" or "wrote:" reply lines
        if re.search(r"writes:|wrote:", line, re.I):
            continue
        cleaned_lines.append(line)

    body = "\n".join(cleaned_lines)

    # Remove emails
    body = re.sub(r"\S+@\S+", " ", body)
    # Remove URLs
    body = re.sub(r"http\S+|www\.\S+", " ", body)
    # Remove HTML tags
    body = re.sub(r"<[^>]+>", " ", body)

    # Remove non-alphanumeric except punctuation
    body = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", body)

    # Collapse multiple spaces/newlines
    body = re.sub(r"\n{2,}", "\n", body)
    body = re.sub(r"\s{2,}", " ", body)

    # Lowercase everything
    body = body.lower().strip()

    # Remove control characters (illegal in Excel/CSV)
    body = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", body)

    return body


def sanitize_for_excel(value: str) -> str:
    """Prevent Excel from misinterpreting text as a formula."""
    if isinstance(value, str) and value and value[0] in ('=', '+', '-', '@'):
        return "'" + value
    return value


def prepare_dataset(dataset_dir, output_csv, max_files=100):
    data_rows = []

    categories = sorted(os.listdir(dataset_dir))
    for category in categories:
        category_path = os.path.join(dataset_dir, category)

        if os.path.isdir(category_path):
            print(f"Processing category: {category}")

            files = sorted(os.listdir(category_path))[:max_files]

            for file_name in files:
                file_path = os.path.join(category_path, file_name)

                try:
                    with open(file_path, "r", encoding="latin1") as f:
                        raw_text = f.read()
                        cleaned_text = clean_body(raw_text)

                        if cleaned_text:  # only keep non-empty text
                            data_rows.append({
                                "file_name": file_name,
                                "category": category,
                                "text": cleaned_text
                            })
                except Exception as e:
                    print(f"⚠️ Skipping {file_name}: {e}")

    df = pd.DataFrame(data_rows, columns=["file_name", "category", "text"])

    # Apply Excel sanitization
    df = df.applymap(sanitize_for_excel)

    # Ensure exactly 2000 rows (100 × 20 categories)
    df = df.head(2000)

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n✅ Dataset created: {output_csv}")
    print(f"📊 Total rows: {len(df)} (should be 2000)")
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    prepare_dataset(DATASET_DIR, OUTPUT_CSV)
