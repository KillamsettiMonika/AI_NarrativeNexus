# app.py
import os
import io
import json
import uuid
import joblib
import requests
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# optional keras import
try:
    from tensorflow.keras.models import load_model as keras_load_model
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False

# ---------- CONFIG ----------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_STORE_JSON = os.path.join(DATA_DIR, "data_store.json")
DATA_STORE_CSV = os.path.join(DATA_DIR, "data_store.csv")

MODELS_DIR = "models"
DATASETS_DIR = "datasets"

# Topic models
LDA_MODEL_PATH = os.path.join(MODELS_DIR, "lda_model.pkl")
LDA_VECT_PATH  = os.path.join(MODELS_DIR, "lda_vectorizer.pkl")
NMF_MODEL_PATH = os.path.join(MODELS_DIR, "nmf_model.pkl")
NMF_VECT_PATH  = os.path.join(MODELS_DIR, "nmf_vectorizer.pkl")

# Sentiment models
RF_PIPELINE_PATH = os.path.join(MODELS_DIR, "amazon_rf_pipeline.pkl")  # or random_forest_model.pkl
LSTM_MODEL_PATH  = os.path.join(MODELS_DIR, "amazon_lstm.h5")
LSTM_TOKENIZER_PATH = os.path.join(MODELS_DIR, "amazon_lstm_tokenizer.pkl")

# Summarization models (pickled pipeline or callable), fallback to simple_textrank
ABSTRACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "abstractive_model.pkl")
EXTRACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "extractive_vectorizer.pkl")

# Evaluation images (if present)
EVAL_IMAGES = {
    "lda": os.path.join(MODELS_DIR, "lda_confusion_matrix.png"),
    "nmf": os.path.join(MODELS_DIR, "nmf_confusion_matrix.png"),
    "rf":  os.path.join(MODELS_DIR, "confusion_matrix_rf.png"),
    "lstm":os.path.join(MODELS_DIR, "confusion_matrix_lstm.png"),
    "extractive_summarization": os.path.join(MODELS_DIR, "evaluation_metrics.png"),
    "abstractive_summarization": os.path.join(MODELS_DIR, "abs_confusion_matrix.png"),
}

NEWS_API_KEY = os.getenv("NEWS_API_KEY", None)

# ---------- HELPERS ----------
def load_json_store(path=DATA_STORE_JSON):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_to_store(records, json_path=DATA_STORE_JSON, csv_path=DATA_STORE_CSV):
    data = load_json_store(json_path)
    data.extend(records)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    df = pd.json_normalize(data)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return len(records)

def safe_read_uploaded(file):
    name = file.name.lower()
    try:
        if name.endswith(".txt"):
            return file.getvalue().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
                pages = [p.extract_text() or "" for p in reader.pages]
                return "\n".join(pages)
            except Exception:
                return ""
        elif name.endswith((".doc", ".docx")):
            try:
                import docx
                doc = docx.Document(io.BytesIO(file.getvalue()))
                return "\n".join(p.text for p in doc.paragraphs)
            except Exception:
                return ""
        else:
            return file.getvalue().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def simple_summarize(text, max_sentences=3):
    import re
    if not text or len(text.split()) < 30:
        return " ".join(text.splitlines()[:3])
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    words = [w.lower() for w in re.findall(r"\w+", text)]
    from collections import Counter
    freq = Counter(words)
    scores = []
    for i, s in enumerate(sents):
        words_in_s = re.findall(r"\w+", s.lower())
        if not words_in_s:
            scores.append((0, i, s))
            continue
        score = sum(freq[w] for w in words_in_s) / len(words_in_s)
        scores.append((score, i, s))
    top = sorted(scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    top_sorted = sorted(top, key=lambda x: x[1])
    return " ".join([t[2].strip() for t in top_sorted])

# ---------- MODEL LOADING ----------
@st.cache_resource
def load_topic_models():
    models = {}
    if os.path.exists(LDA_MODEL_PATH) and os.path.exists(LDA_VECT_PATH):
        try:
            models['LDA'] = (joblib.load(LDA_MODEL_PATH), joblib.load(LDA_VECT_PATH))
        except Exception:
            pass
    if os.path.exists(NMF_MODEL_PATH) and os.path.exists(NMF_VECT_PATH):
        try:
            models['NMF'] = (joblib.load(NMF_MODEL_PATH), joblib.load(NMF_VECT_PATH))
        except Exception:
            pass
    return models

@st.cache_resource
def load_sentiment_models():
    models = {}
    if os.path.exists(RF_PIPELINE_PATH):
        try:
            models['RF'] = joblib.load(RF_PIPELINE_PATH)
        except Exception:
            pass
    if HAS_KERAS and os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_TOKENIZER_PATH):
        try:
            lstm = keras_load_model(LSTM_MODEL_PATH)
            tokenizer = joblib.load(LSTM_TOKENIZER_PATH)
            models['LSTM'] = (lstm, tokenizer)
        except Exception:
            pass
    return models

@st.cache_resource
def load_summarizer_models():
    s = {}
    if os.path.exists(ABSTRACTIVE_MODEL_PATH):
        try:
            s['Abstractive'] = joblib.load(ABSTRACTIVE_MODEL_PATH)
        except Exception:
            s['Abstractive_error'] = str(traceback.format_exc())[:1000]
    if os.path.exists(EXTRACTIVE_MODEL_PATH):
        try:
            s['Extractive'] = joblib.load(EXTRACTIVE_MODEL_PATH)
        except Exception:
            s['Extractive_error'] = str(traceback.format_exc())[:1000]
    return s

topic_models = load_topic_models()
sentiment_models = load_sentiment_models()
summarizer_models = load_summarizer_models()

# ---------- PREDICTION HELPERS ----------
def predict_topic_single(text, model_name):
    if model_name not in topic_models:
        return None, "model unavailable"
    model, vect = topic_models[model_name]
    X = vect.transform([text])
    try:
        doc_topic = model.transform(X)
        top = doc_topic.argmax(axis=1)[0]
        # try to produce human friendly text using top words if available
        top_words = []
        try:
            feature_names = vect.get_feature_names_out()
            if hasattr(model, "components_"):
                comp = model.components_[top]
                top_idx = comp.argsort()[::-1][:10]
                top_words = [feature_names[i] for i in top_idx]
        except Exception:
            pass
        label = f"Topic {top}"
        if top_words:
            label = f"Topic {top} — " + ", ".join(top_words[:6])
        return label, None
    except Exception as e:
        return None, str(e)

def predict_sentiment_single(text, model_name):
    if model_name not in sentiment_models:
        return None, "model unavailable"
    if model_name == "LSTM":
        lstm, tokenizer = sentiment_models['LSTM']
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=200)
        p = (lstm.predict(seq) > 0.5).astype(int).flatten()[0]
        return "positive" if int(p) == 1 else "negative", None
    else:
        m = sentiment_models[model_name]
        try:
            p = m.predict([text])[0]
            if isinstance(p, (int, np.integer, float, np.floating)):
                return "positive" if int(p) == 1 else "negative", None
            return str(p), None
        except Exception as e:
            return None, str(e)

def summarize_text(text, method):
    if method == "simple_textrank":
        return simple_summarize(text)
    if method == "Abstractive":
        model = summarizer_models.get("Abstractive")
        if model is None:
            return None
        # If model is a huggingface pipeline pickled, it should be callable
        try:
            if callable(model):
                out = model(text, max_length=130, min_length=30, do_sample=False)
                if isinstance(out, list) and 'summary_text' in out[0]:
                    return out[0]['summary_text']
                # fallback if out is a string
                return out if isinstance(out, str) else str(out)
            # else maybe model is an object with .summarize
            if hasattr(model, "summarize"):
                return model.summarize(text)
        except Exception:
            return None
    if method == "Extractive":
        model = summarizer_models.get("Extractive")
        if model is None:
            return None
        try:
            # If vectorizer + extraction pipeline saved, apply transform then simple top-n sentences heuristic
            if hasattr(model, "transform"):
                vec = model
                # fallback: use simple summarizer (extract first 3 sentences)
                sents = text.split(".")
                return ". ".join(sents[:3]).strip()
        except Exception:
            return None
    return None

# ---------- UI ----------
st.set_page_config(page_title="NarrativeNexus", layout="wide")
st.title("📌 AI Narrative Nexus")

# Sidebar navigation
pages = [
    "Home", "Topic Modeling", "Sentiment Analysis", "Text Summarization",
    "Data Visualization", "Evaluation & Analysis", "Live Demo", "About"
]
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("Go to:", pages, index=0)

# Shared model selection widgets (kept on the relevant pages)
def input_source_widget(prefix=""):
    st.markdown("**Select Input Type**")
    input_type = st.radio("Input source:", ["Free Text", "Reddit URL", "News API"], index=0, key=prefix+"input_type")
    if input_type == "Free Text":
        text = st.text_area("Paste the paragraph / article / content to analyze:", height=180, key=prefix+"free_text")
    elif input_type == "Reddit URL":
        url = st.text_input("Paste Reddit post URL (will attempt to extract):", key=prefix+"reddit_url")
        text = None
        if st.button("Fetch Reddit content", key=prefix+"fetch_reddit"):
            try:
                # very simple HTML fetch - works for many reddit pages; not robust
                res = requests.get(url, headers={"User-Agent": "NarrativeNexus/0.1"})
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(res.text, "html.parser")
                # collect main text from common selectors
                candidates = soup.find_all(["p"])
                text = " ".join(p.get_text(" ", strip=True) for p in candidates)
                st.success("Fetched content (raw). Please scroll down to run model.")
            except Exception as e:
                st.error(f"Could not fetch Reddit URL: {e}")
                text = ""
    else:  # News API
        query = st.text_input("Enter News query (or topic):", key=prefix+"news_query")
        text = None
        if st.button("Fetch top article", key=prefix+"fetch_news"):
            if NEWS_API_KEY is None:
                st.error("No NEWS_API_KEY configured as environment variable.")
                text = ""
            else:
                try:
                    params = {"q": query, "pageSize": 1, "apiKey": NEWS_API_KEY}
                    r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=12)
                    j = r.json()
                    if j.get("articles"):
                        art = j["articles"][0]
                        text = (art.get("title","") or "") + ". " + (art.get("description","") or "") + ". " + (art.get("content","") or "")
                        st.success("Fetched top article content.")
                    else:
                        st.warning("No articles found.")
                        text = ""
                except Exception as e:
                    st.error(f"News fetch failed: {e}")
                    text = ""
    return input_type, locals().get("text", None)

def show_record_json(record):
    st.markdown("**Result (JSON)**")
    st.json(record)

# ---------- PAGE: Home ----------
if page == "Home":
    st.header("About AI Narrative Nexus")
    st.markdown("""
    **AI Narrative Nexus** converts unstructured text to insights using:
    - Topic Modeling (LDA / NMF)
    - Sentiment Analysis (Random Forest / LSTM)
    - Text Summarization (Abstractive / Extractive / Textrank)
    - Visualizations & Evaluation (confusion matrices, EDA plots)

    **Project structure (expected)**:
    - `models/` — pre-trained models (pkl / h5)
    - `datasets/` — datasets used for model training (CSV or folder)
    - `data/` — runtime data store for predictions

    **How to use**
    - Choose a page on the left
    - Pick model, paste text / URL / fetch via News API
    - Run and inspect results — predictions are saved to `data/data_store.json`

    > Tip: If a model is missing, add the trained model files to `models/` with the names used in the app.
    """)
    st.info("Model availability:\n\nTopic models: " + ", ".join(topic_models.keys() or ["(none)"]))
    st.info("Sentiment models: " + ", ".join(sentiment_models.keys() or ["(none)"]))
    st.info("Summarizers: " + ", ".join([k for k in summarizer_models.keys() if not k.endswith('_error')] or ["(none)"]))

# ---------- PAGE: Topic Modeling ----------
elif page == "Topic Modeling":
    st.header("🧩 Topic Modeling")
    st.markdown("Choose a topic model and input source. The app will show a predicted topic (top words) and save the result.")

    model_choice = st.selectbox("Select Topic Model", ["none", "LDA", "NMF"], index=0 if not topic_models else (1 if "LDA" in topic_models else 0))
    input_type, text = input_source_widget(prefix="topic_")

    if st.button("Run Topic Modeling"):
        if model_choice == "none":
            st.error("Select a topic model first.")
        else:
            if not text:
                st.warning("No text available to analyze.")
            else:
                with st.spinner("Predicting topic..."):
                    pred, err = predict_topic_single(text, model_choice)
                if err:
                    st.error(f"Topic prediction error: {err}")
                else:
                    st.success(f"Topic predicted: {pred}")
                record = {
                    "id": str(uuid.uuid4()),
                    "source": input_type,
                    "model": model_choice,
                    "input": text,
                    "topic_prediction": pred,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                save_to_store([record])
                show_record_json(record)

# ---------- PAGE: Sentiment Analysis ----------
elif page == "Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    st.markdown("Choose a sentiment model (Random Forest or LSTM) and provide input.")

    sent_choice = st.selectbox("Select Sentiment Model", ["none"] + list(sentiment_models.keys()))
    input_type, text = input_source_widget(prefix="sent_")

    if st.button("Run Sentiment Analysis"):
        if sent_choice == "none":
            st.error("Choose a sentiment model.")
        else:
            if not text:
                st.warning("No text available.")
            else:
                with st.spinner("Running sentiment model..."):
                    pred, err = predict_sentiment_single(text, sent_choice)
                if err:
                    st.error(f"Error: {err}")
                else:
                    if pred == "positive":
                        st.success("Positive 🙂")
                    elif pred == "negative":
                        st.error("Negative 🙁")
                    else:
                        st.info(str(pred))
                record = {
                    "id": str(uuid.uuid4()),
                    "source": input_type,
                    "model": sent_choice,
                    "input": text,
                    "predicted_sentiment": pred,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                save_to_store([record])
                show_record_json(record)

# ---------- PAGE: Text Summarization ----------
elif page == "Text Summarization":
    st.header("✂️ Text Summarization")
    st.markdown("Select summarizer and input text (or fetch from News/Reddit).")

    summarizer_choice = st.selectbox("Summarizer", ["simple_textrank", "Abstractive", "Extractive"])
    input_type, text = input_source_widget(prefix="summ_")

    if st.button("Generate Summary"):
        if not text:
            st.warning("No text provided.")
        else:
            with st.spinner("Generating summary..."):
                summary = summarize_text(text, summarizer_choice)
            if summary:
                st.success(f"Summary generated using {summarizer_choice}!")
                st.markdown("### ✨ Summary:")
                st.write(summary)
            else:
                st.error("Could not generate summary with selected model. Falling back to TextRank.")
                fallback = simple_summarize(text)
                st.write(fallback)

            record = {
                "id": str(uuid.uuid4()),
                "source": input_type,
                "model": summarizer_choice,
                "input": text,
                "summary": summary or fallback,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            save_to_store([record])
            st.expander("View saved record (json)").json(record)

# ---------- PAGE: Data Visualization ----------
elif page == "Data Visualization":
    st.title("📊 Data Visualization (EDA)")
    model_area = st.selectbox("Choose area", ["Topic Modeling", "Sentiment Analysis", "Text Summarization"])

    if model_area == "Topic Modeling":
        st.header("🧩 Topic Modeling - Exploratory Analysis")

        # Path to 20 Newsgroups dataset
        base_path = os.path.join(DATASETS_DIR, "20news-18828-20251028T113358Z-1-001/20news-18828")
        categories = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        topic_counts = {cat: len(os.listdir(os.path.join(base_path, cat))) for cat in categories}

        df_topics = pd.DataFrame(list(topic_counts.items()), columns=["Category", "Documents"])
        st.subheader("📈 Documents per Category")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_topics.sort_values("Documents", ascending=False), x="Documents", y="Category", palette="mako")
        st.pyplot(fig)

        st.subheader("🥧 Topic Distribution (Pie Chart)")
        fig1, ax1 = plt.subplots()
        ax1.pie(df_topics["Documents"], labels=df_topics["Category"], autopct="%1.1f%%", startangle=140)
        st.pyplot(fig1)

        # Simulate text length data (replace with real text lengths if you want)
        import numpy as np
        df_topics["Text_Length"] = np.random.randint(300, 1200, size=len(df_topics))

        st.subheader("📦 Boxplot of Text Lengths by Category")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df_topics, x="Category", y="Text_Length", ax=ax2)
        plt.xticks(rotation=90)
        st.pyplot(fig2)

        st.subheader("🎻 Violin Plot - Text Length Distribution per Topic")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        sns.violinplot(data=df_topics, x="Category", y="Text_Length", inner="quart", ax=ax3)
        plt.xticks(rotation=90)
        st.pyplot(fig3)

        # Generate mock top words and frequencies
        words = ["data", "science", "religion", "sports", "politics", "space", "hardware", "software", "crypt", "windows"]
        freq = np.random.randint(50, 400, size=len(words))
        df_words = pd.DataFrame({"Word": words, "Frequency": freq})

        st.subheader("🔤 Top Word Frequencies Across Topics")
        fig4, ax4 = plt.subplots()
        sns.barplot(data=df_words.sort_values("Frequency", ascending=False), x="Frequency", y="Word", ax=ax4)
        st.pyplot(fig4)

        st.subheader("🔗 Top Bigram Frequencies (Mock Example)")
        bigrams = ["data science", "religious belief", "space research", "sports news", "political view"]
        bigram_freq = np.random.randint(20, 100, size=len(bigrams))
        df_bigrams = pd.DataFrame({"Bigram": bigrams, "Frequency": bigram_freq})
        fig5, ax5 = plt.subplots()
        sns.barplot(data=df_bigrams.sort_values("Frequency", ascending=False), x="Frequency", y="Bigram", ax=ax5)
        st.pyplot(fig5)

        st.subheader("📉 Scatter Plot — Topic Index vs Text Length")
        df_topics["Topic Index"] = range(len(df_topics))
        fig6, ax6 = plt.subplots()
        sns.scatterplot(data=df_topics, x="Topic Index", y="Text_Length", hue="Category", s=80)
        st.pyplot(fig6)

        st.success("✅ Visualization generated for Topic Modeling dataset")

    elif model_area == "Sentiment Analysis":
        # keep your previous sentiment plots
        st.header("❤️ Sentiment Analysis Visualizations")
        st.info("Bar, Pie, and Box plots for sentiment data are shown here.")
        labels = ["Positive", "Negative"]
        counts = [250, 250]
        fig1, ax1 = plt.subplots()
        ax1.bar(labels, counts)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.pie(counts, labels=labels, autopct="%1.1f%%")
        st.pyplot(fig2)

    elif model_area == "Text Summarization":
        st.header("🧠 Text Summarization Visualizations")
        st.info("Displays relationships between original and summary lengths.")
        data = pd.DataFrame({
            "Original Length": [1000, 800, 1200, 950, 1100],
            "Summary Length": [200, 180, 250, 210, 220]
        })
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x="Original Length", y="Summary Length", ax=ax)
        st.pyplot(fig)
        st.line_chart(data)


# ---------- PAGE: Evaluation & Analysis ----------
elif page == "Evaluation & Analysis":
    st.header("📈 Evaluation & Analysis")
    st.markdown("View confusion matrices and evaluation artifacts for each model if available.")
    # list available images from EVAL_IMAGES
    for key, path in EVAL_IMAGES.items():
        st.markdown(f"**{key.upper()}**")
        if os.path.exists(path):
            st.image(path, caption=os.path.basename(path), use_column_width=True)
        else:
            st.warning(f"Confusion matrix / image not found for: {os.path.basename(path)}")

    # show a summary of saved predictions
    st.subheader("Saved predictions (data store)")
    records = load_json_store()
    if records:
        df = pd.json_normalize(records)
        st.dataframe(df.tail(100))
        st.download_button("Download predictions CSV", DATA_STORE_CSV, file_name="data_store.csv")
    else:
        st.info("No saved predictions yet (run the analyzers to generate).")

# ---------- PAGE: Live Demo (blank / placeholder) ----------
elif page == "Live Demo":
    st.header("Live Demo")
    st.info("This page is reserved for live demonstrations. (Blank placeholder).")

# ---------- PAGE: About ----------
elif page == "About":
    st.header("About & Team")
    st.markdown("""
    **AI Narrative Nexus** — Project for Infosys Internship (demo).
    - Team: Your name(s)
    - Contact: (add your email)
    - Description: A web app to perform topic modelling, sentiment analysis, and text summarization and visualize results.

    Replace the placeholder models under `models/` with your trained models to enable full functionality.
    """)
    st.markdown("**Project files**")
    st.write(sorted(os.listdir(".")))

# ---------- END ----------
