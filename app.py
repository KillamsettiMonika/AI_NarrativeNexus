# app.py
import os
import io
import uuid
import json
import joblib
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timezone

# try to import the project's preprocessing function (adjust if your path differs)
try:
    from topicmodelling.pre_processing.text_processing import preprocess_series
except Exception:
    def preprocess_series(xs):
        return pd.Series(xs).astype(str).str.lower().tolist()

# attempt keras import for LSTM use
try:
    from tensorflow.keras.models import load_model as keras_load_model
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False

# ----------------------------
# Configuration
# ----------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_STORE_JSON = os.path.join(DATA_DIR, "data_store.json")
DATA_STORE_CSV = os.path.join(DATA_DIR, "data_store.csv")

MODELS_DIR = "models"
LDA_MODEL_PATH = os.path.join(MODELS_DIR, "lda_model.pkl")
LDA_VECT_PATH = os.path.join(MODELS_DIR, "lda_vectorizer.pkl")
NMF_MODEL_PATH = os.path.join(MODELS_DIR, "nmf_model.pkl")
NMF_VECT_PATH = os.path.join(MODELS_DIR, "nmf_vectorizer.pkl")
RF_PIPELINE_PATH = os.path.join(MODELS_DIR, "amazon_rf_pipeline.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.h5")
LSTM_TOKENIZER_PATH = os.path.join(MODELS_DIR, "lstm_tokenizer.pkl")
CLASSIFIER_PIPELINE_PATH = os.path.join(MODELS_DIR, "text_classifier.pkl")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", None)

# ----------------------------
# Utility functions
# ----------------------------
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

# ----------------------------
# Model loaders
# ----------------------------
@st.cache_resource
def load_topic_models():
    models = {}
    if os.path.exists(LDA_MODEL_PATH) and os.path.exists(LDA_VECT_PATH):
        try:
            lda = joblib.load(LDA_MODEL_PATH)
            vect = joblib.load(LDA_VECT_PATH)
            models['lda'] = (lda, vect)
        except Exception:
            pass
    if os.path.exists(NMF_MODEL_PATH) and os.path.exists(NMF_VECT_PATH):
        try:
            nmf = joblib.load(NMF_MODEL_PATH)
            vect = joblib.load(NMF_VECT_PATH)
            models['nmf'] = (nmf, vect)
        except Exception:
            pass
    return models

@st.cache_resource
def load_sentiment_models():
    models = {}
    if os.path.exists(RF_PIPELINE_PATH):
        try:
            rf = joblib.load(RF_PIPELINE_PATH)
            models['rf'] = rf
        except Exception:
            pass
    if HAS_KERAS and os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_TOKENIZER_PATH):
        try:
            lstm = keras_load_model(LSTM_MODEL_PATH)
            tokenizer = joblib.load(LSTM_TOKENIZER_PATH)
            models['lstm'] = (lstm, tokenizer)
        except Exception:
            pass
    if os.path.exists(CLASSIFIER_PIPELINE_PATH):
        try:
            pipe = joblib.load(CLASSIFIER_PIPELINE_PATH)
            models['pipeline'] = pipe
        except Exception:
            pass
    return models

topic_models = load_topic_models()
sentiment_models = load_sentiment_models()

# ----------------------------
# Prediction Helpers
# ----------------------------
def predict_topic(texts, model_choice='lda'):
    if model_choice not in topic_models:
        return None, "Topic model not available"
    model, vect = topic_models[model_choice]
    X = vect.transform(texts)
    doc_topic = model.transform(X)
    topics = doc_topic.argmax(axis=1).tolist()
    return topics, None

def predict_sentiment(texts, model_choice='rf'):
    if model_choice not in sentiment_models:
        return None, "Sentiment model not available"
    if model_choice == 'lstm':
        lstm, tokenizer = sentiment_models['lstm']
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seqs = tokenizer.texts_to_sequences(texts)
        seqs = pad_sequences(seqs, maxlen=200)
        preds = (lstm.predict(seqs) > 0.5).astype(int).flatten().tolist()
        return ["positive" if p == 1 else "negative" for p in preds], None
    else:
        model = sentiment_models[model_choice]
        preds = model.predict(texts)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        mapped = []
        for p in preds:
            if isinstance(p, (int, float)):
                mapped.append("positive" if int(p) == 1 else "negative")
            else:
                mapped.append(str(p))
        return mapped, None

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="NarrativeNexus", layout="wide")
st.title("📌 NarrativeNexus — Collect, Topic & Sentiment")

st.markdown("Input text (link / news query / free text / file). Choose topic & sentiment models and press **Predict**.")

input_col, model_col, action_col = st.columns([2, 1, 1])

with input_col:
    input_type = st.radio("Input type", ["Free Text", "Upload document", "News query (text)", "Reddit Post (URL)"])
    user_input = st.text_input("Provide text / query / URL")
    uploaded = None
    if input_type == "Upload document":
        uploaded = st.file_uploader("Upload txt/pdf/docx", type=["txt", "pdf", "doc", "docx"])

with model_col:
    st.subheader("Model selection")
    topic_options = ["none"] + list(topic_models.keys())
    selected_topic_model = st.selectbox("Topic model", topic_options, index=0)
    sent_options = ["none"] + list(sentiment_models.keys())
    selected_sentiment_model = st.selectbox("Sentiment model", sent_options, index=0)
    summarizer_choice = st.selectbox("Summarizer", ["simple_textrank"], index=0)

with action_col:
    st.subheader("Actions")
    predict_btn = st.button("🔮 Predict & Save")
    if st.checkbox("Show raw model availability"):
        st.write("Topic models:", list(topic_models.keys()))
        st.write("Sentiment models:", list(sentiment_models.keys()))
        st.write("Keras available:", HAS_KERAS)

# ----------------------------
# On Predict
# ----------------------------
if predict_btn:
    st.info("Running pipeline...")
    try:
        # --- Input handling ---
        if input_type == "Free Text":
            text = user_input
            record = {"id": str(uuid.uuid4()), "source": "manual", "author": "user",
                      "timestamp": datetime.now(timezone.utc).isoformat(),
                      "text": text, "metadata": {}}
        elif input_type == "Upload document":
            if uploaded is None:
                st.error("Please upload a file.")
                st.stop()
            text = safe_read_uploaded(uploaded)
            record = {"id": str(uuid.uuid4()), "source": "upload", "author": "user",
                      "timestamp": datetime.now(timezone.utc).isoformat(),
                      "text": text, "metadata": {"filename": uploaded.name}}
        else:
            text = user_input
            record = {"id": str(uuid.uuid4()), "source": input_type, "author": "user",
                      "timestamp": datetime.now(timezone.utc).isoformat(),
                      "text": text, "metadata": {}}

        # --- Preprocess ---
        raw_texts = [record["text"]]
        try:
            cleaned = preprocess_series(raw_texts)
            cleaned_list = cleaned.tolist() if isinstance(cleaned, pd.Series) else list(cleaned)
        except Exception:
            cleaned_list = [str(t).lower() for t in raw_texts]

        # --- Predictions ---
        topic_out = None
        if selected_topic_model != "none":
            topics, _ = predict_topic(cleaned_list, model_choice=selected_topic_model)
            topic_out = topics[0] if topics else None

        sentiment_out = None
        if selected_sentiment_model != "none":
            sents, _ = predict_sentiment(cleaned_list, model_choice=selected_sentiment_model)
            sentiment_out = sents[0] if sents else None

        summary = simple_summarize(record["text"], max_sentences=3)

        # --- Result ---
        record_result = {**record, "clean_text": cleaned_list[0],
                         "predicted_topic": str(topic_out) if topic_out else None,
                         "predicted_sentiment": str(sentiment_out) if sentiment_out else None,
                         "summary": summary, "predicted_at": datetime.now(timezone.utc).isoformat()}

        save_to_store([record_result])
        st.success("✅ Prediction complete and saved!")

        # ----------------------------
        # Results Display (extra details)
        # ----------------------------
        st.header("📝 Results")

        st.subheader("🔮 Predicted Topic")
        st.success(record_result["predicted_topic"] or "— (topic model not available)")

        st.subheader("😊 Sentiment")
        if record_result["predicted_sentiment"]:
            if record_result["predicted_sentiment"].lower() == "positive":
                st.success("Positive 🙂")
            elif record_result["predicted_sentiment"].lower() == "negative":
                st.error("Negative 🙁")
            else:
                st.info(record_result["predicted_sentiment"])
        else:
            st.write("— (sentiment model not available)")

        st.subheader("📄 Summary")
        st.write(record_result["summary"] or "(no summary)")

        st.subheader("🔍 Text Details")
        with st.expander("Show Original Text"):
            st.write(record_result["text"][:2000])

        with st.expander("Show Cleaned Text"):
            st.write(record_result["clean_text"][:2000] if record_result["clean_text"] else "")

        st.subheader("📂 Metadata")
        st.json(record_result.get("metadata", {}))

        if st.checkbox("Show Full JSON Record"):
            st.json(record_result)

    except Exception as e:
        st.error(f"Unexpected error: {e}")
