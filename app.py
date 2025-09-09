import os
import uuid
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timezone
import praw

# Load environment variables
load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Reddit instance
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Function to fetch Reddit post
def fetch_reddit_post(url):
    submission = reddit.submission(url=url)
    record = {
        "id": str(uuid.uuid4()),
        "source": "reddit",
        "author": submission.author.name if submission.author else "unknown",
        "timestamp": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
        # keep emojis and full text
        "text": submission.title + "\n" + submission.selftext,
        "metadata": {
            "language": "en",
            "likes": submission.score,
            "rating": None,
            "url": url
        }
    }
    return record

# Function to fetch News Article
def fetch_news(query):
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

# Save data to file
def save_data(records, format_choice):
    df = pd.json_normalize(records)
    if format_choice == "CSV":
        df.to_csv("output_data.csv", index=False, encoding="utf-8")
        return "✅ Data saved to output_data.csv"
    else:
        with open("output_data.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        return "✅ Data saved to output_data.json"

# Streamlit UI
st.title("📌 NarrativeNexus Data Collector")
st.write("Paste a Reddit post link or News query. Data will be collected and saved to CSV/JSON.")

option = st.radio("Choose Source:", ["Reddit Post", "News Article"])
user_input = st.text_input("Enter Reddit link or News query:")
save_format = st.selectbox("Save as:", ["CSV", "JSON"])

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
            message = save_data(records, save_format)
            st.success(message)
            st.subheader("Preview")
            st.json(records[0])

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
