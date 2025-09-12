import os
import uuid
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timezone
import praw

# ==============================
# Setup
# ==============================
load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Ensure data folder exists
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE_JSON = os.path.join(DATA_DIR, "data_store.json")
DATA_FILE_CSV = os.path.join(DATA_DIR, "data_store.csv")

# Reddit instance
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# ==============================
# Functions
# ==============================

def fetch_reddit_post(url):
    """Fetch a single Reddit post"""
    submission = reddit.submission(url=url)
    record = {
        "id": str(uuid.uuid4()),
        "source": "reddit",
        "author": submission.author.name if submission.author else "unknown",
        "timestamp": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
        "text": (submission.title or "") + "\n" + (submission.selftext or ""),
        "metadata": {
            "language": "en",
            "likes": submission.score,
            "rating": None,
            "url": url
        }
    }
    return record


def fetch_news(query):
    """Fetch first News article"""
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
            "likes": None,
            "rating": None,
            "url": article.get("url")
        }
    }
    return record


def load_data():
    """Load existing JSON dataset"""
    if os.path.exists(DATA_FILE_JSON):
        with open(DATA_FILE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_data(new_records):
    """Save new data (append to dataset)"""
    data = load_data()
    data.extend(new_records)   # append new records

    # Save JSON
    with open(DATA_FILE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Save CSV
    df = pd.json_normalize(data)
    df.to_csv(DATA_FILE_CSV, index=False, encoding="utf-8-sig")

    return f"✅ Data saved to {DATA_FILE_JSON} and {DATA_FILE_CSV}"


# ==============================
# Streamlit UI
# ==============================
st.title("📌 NarrativeNexus Data Collector")
st.write("Paste a Reddit post link or News query. Data will be collected and stored in `data/` folder.")

option = st.radio("Choose Source:", ["Reddit Post", "News Article"])
user_input = st.text_input("Enter Reddit link or News query:")

if st.button("Fetch & Save"):
    records = []
    try:
        if option == "Reddit Post":
            record = fetch_reddit_post(user_input)
            records.append(record)
        elif option == "News Article":
            record = fetch_news(user_input)
            if record:
                records.append(record)
            else:
                st.error("❌ No news found for this query.")

        if records:
            message = save_data(records)
            st.success(message)
            st.subheader("Preview")
            st.json(records[0])

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
