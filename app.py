import os
import uuid
import json
import requests
import pandas as pd
import streamlit as st
import docx
import pdfplumber
from dotenv import load_dotenv
from datetime import datetime, timezone  
import praw

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "data_store.json")

# ---------------------------
# Utility functions
# ---------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_data(new_record):
    data = load_data()
    data.append(new_record)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def fetch_reddit_post(url):
    """Fetch a single Reddit post given its URL (emojis preserved)"""
    submission = reddit.submission(url=url)
    record = {
        "id": str(uuid.uuid4()),
        "source": "reddit",
        "author": submission.author.name if submission.author else "unknown",
        "timestamp": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
        "text": submission.title + "\n" + submission.selftext,
        "metadata": {
            "language": "en",
            "likes": submission.score,
            "url": url
        }
    }
    return record

def fetch_news(query):
    """Fetch first News article from NewsAPI matching a query"""
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()

    if "articles" not in response or len(response["articles"]) == 0:
        return None

    article = response["articles"][0]
    record = {
        "id": str(uuid.uuid4()),
        "source": "news",
        "author": article.get("author") or "unknown",
        "timestamp": article.get("publishedAt"),
        "text": (article.get("title") or "") + "\n" + (article.get("description") or ""),
        "metadata": {
            "language": "en",
            "url": article.get("url")
        }
    }
    return record

def read_txt(file):
    return file.read().decode("utf-8")

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def create_record(filename, source_type, file_type, content):
    return {
        "id": int(datetime.now().timestamp()),
        "filename": filename,
        "source_type": source_type,
        "file_type": file_type,
        "content": content,
        "upload_time": datetime.now().isoformat()
    }

# ---------------------------
# Streamlit App
# ---------------------------
st.title("📰 NarrativeNexus Data Collector")

st.write("Choose input type: Reddit Post, News Article, Upload a file, or Paste text.")

option = st.radio("Choose Source:", ["Reddit Post", "News Article", "Upload File", "Paste Text"])
save_format = st.selectbox("Save as:", ["CSV", "JSON"])
records = []

if option == "Reddit Post":
    url = st.text_input("Enter Reddit Post URL:")
    if st.button("Fetch Reddit Post"):
        try:
            record = fetch_reddit_post(url)
            records.append(record)
            save_data(record)
            st.success("Reddit post saved!")
            st.json(record)
        except Exception as e:
            st.error(f"Error: {e}")

elif option == "News Article":
    query = st.text_input("Enter News Query:")
    if st.button("Fetch News"):
        try:
            record = fetch_news(query)
            if record:
                records.append(record)
                save_data(record)
                st.success("News article saved!")
                st.json(record)
            else:
                st.error("No news found.")
        except Exception as e:
            st.error(f"Error: {e}")

elif option == "Upload File":
    uploaded_file = st.file_uploader("Upload a file (.txt, .csv, .docx, .pdf)", type=["txt", "csv", "docx", "pdf"])
    if uploaded_file is not None:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == ".txt":
                content = read_txt(uploaded_file)
            elif file_extension == ".csv":
                content = read_csv(uploaded_file)
            elif file_extension == ".docx":
                content = read_docx(uploaded_file)
            elif file_extension == ".pdf":
                content = read_pdf(uploaded_file)
            else:
                content = None
                st.error("Unsupported file type!")

            if content:
                record = create_record(uploaded_file.name, "file", file_extension, content)
                save_data(record)
                st.success("File uploaded & stored!")
                st.write(content[:1000] + "..." if len(content) > 1000 else content)
        except Exception as e:
            st.error(f"Error: {e}")

elif option == "Paste Text":
    text_input = st.text_area("Paste your text here:")
    if st.button("Save Text"):
        if text_input.strip() != "":
            record = create_record("pasted_text", "pasted", "raw", text_input)
            save_data(record)
            st.success("Text saved!")
            st.write(text_input[:1000] + "..." if len(text_input) > 1000 else text_input)
        else:
            st.error("No text entered!")

# ---------------------------
# Export Records
# ---------------------------
if st.button("Export All Records"):
    data = load_data()
    if save_format == "CSV":
        df = pd.json_normalize(data)
        df.to_csv("output_data.csv", index=False, encoding="utf-8-sig")
        st.success("All data exported to output_data.csv")
    else:
        with open("output_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        st.success("All data exported to output_data.json")
