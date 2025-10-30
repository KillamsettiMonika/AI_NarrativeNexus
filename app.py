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

# ----------------------------
# Optional preprocessing import
# ----------------------------
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
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "amazon_lstm.h5")
LSTM_TOKENIZER_PATH = os.path.join(MODELS_DIR, "amazon_lstm_tokenizer.pkl")
CLASSIFIER_PIPELINE_PATH = os.path.join(MODELS_DIR, "text_classifier.pkl")

# summarization model paths
ABSTRACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "abstractive_model.pkl")
EXTRACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "extractive_vectorizer.pkl")

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
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        elif name.endswith((".doc", ".docx")):
            import docx
            doc = docx.Document(io.BytesIO(file.getvalue()))
            return "\n".join(p.text for p in doc.paragraphs)
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
        lda = joblib.load(LDA_MODEL_PATH)
        vect = joblib.load(LDA_VECT_PATH)
        models['lda'] = (lda, vect)
    if os.path.exists(NMF_MODEL_PATH) and os.path.exists(NMF_VECT_PATH):
        nmf = joblib.load(NMF_MODEL_PATH)
        vect = joblib.load(NMF_VECT_PATH)
        models['nmf'] = (nmf, vect)
    return models

@st.cache_resource
def load_sentiment_models():
    models = {}
    if os.path.exists(RF_PIPELINE_PATH):
        models['rf'] = joblib.load(RF_PIPELINE_PATH)
    if HAS_KERAS and os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_TOKENIZER_PATH):
        lstm = keras_load_model(LSTM_MODEL_PATH)
        tokenizer = joblib.load(LSTM_TOKENIZER_PATH)
        models['lstm'] = (lstm, tokenizer)
    if os.path.exists(CLASSIFIER_PIPELINE_PATH):
        models['pipeline'] = joblib.load(CLASSIFIER_PIPELINE_PATH)
    return models

@st.cache_resource
def load_summarizer_models():
    summarizers = {}
    if os.path.exists(ABSTRACTIVE_MODEL_PATH):
        try:
            abstractive = joblib.load(ABSTRACTIVE_MODEL_PATH)
            summarizers["abstractive"] = abstractive
        except Exception as e:
            print(f"⚠️ Error loading abstractive model: {e}")
    if os.path.exists(EXTRACTIVE_MODEL_PATH):
        try:
            extractive = joblib.load(EXTRACTIVE_MODEL_PATH)
            summarizers["extractive"] = extractive
        except Exception as e:
            print(f"⚠️ Error loading extractive model: {e}")
    return summarizers

topic_models = load_topic_models()
sentiment_models = load_sentiment_models()
summarizer_models = load_summarizer_models()

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
        mapped = ["positive" if int(p) == 1 else "negative" for p in preds]
        return mapped, None

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="NarrativeNexus", layout="wide")
st.title("📌 NarrativeNexus — Collect, Topic, Sentiment & Summarization")

input_col, model_col, action_col = st.columns([2, 1, 1])

with input_col:
    input_type = st.radio("Input type", ["Free Text", "Upload document", "News query (text)", "Reddit Post (URL)"])
    user_input = st.text_input("Provide text / query / URL")
    uploaded = st.file_uploader("Upload txt/pdf/docx", type=["txt", "pdf", "doc", "docx"]) if input_type == "Upload document" else None

with model_col:
    st.subheader("Model selection")
    topic_options = ["none"] + list(topic_models.keys())
    selected_topic_model = st.selectbox("Topic model", topic_options, index=0)
    sent_options = ["none"] + list(sentiment_models.keys())
    selected_sentiment_model = st.selectbox("Sentiment model", sent_options, index=0)
    summarizer_options = ["simple_textrank"] + list(summarizer_models.keys())
    summarizer_choice = st.selectbox("Summarizer", summarizer_options, index=0)

with action_col:
    st.subheader("Actions")
    predict_btn = st.button("🔮 Predict & Save")
    if st.checkbox("Show model availability"):
        st.write("Topic models:", list(topic_models.keys()))
        st.write("Sentiment models:", list(sentiment_models.keys()))
        st.write("Summarizers:", list(summarizer_models.keys()))

# ----------------------------
# On Predict
# ----------------------------
if predict_btn:
    st.info("Running pipeline...")
    try:
        # Input handling
        if input_type == "Free Text":
            text = user_input
        elif input_type == "Upload document" and uploaded is not None:
            text = safe_read_uploaded(uploaded)
        else:
            text = user_input

        record = {
            "id": str(uuid.uuid4()),
            "source": input_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": text,
        }

        # Preprocess
        cleaned_list = preprocess_series([text])

        # Predictions
        topic_out = None
        if selected_topic_model != "none":
            topics, _ = predict_topic(cleaned_list, selected_topic_model)
            topic_out = topics[0] if topics else None

        sentiment_out = None
        if selected_sentiment_model != "none":
            sents, _ = predict_sentiment(cleaned_list, selected_sentiment_model)
            sentiment_out = sents[0] if sents else None

        # Summarization
        summary = None
        if summarizer_choice == "simple_textrank":
            summary = simple_summarize(text)
        elif summarizer_choice == "abstractive":
            model = summarizer_models.get("abstractive")
            if model:
                summary = model(text)[0]['summary_text'] if callable(model) else "(invalid abstractive model)"
        elif summarizer_choice == "extractive":
            model = summarizer_models.get("extractive")
            if model:
                summary = model.transform([text])[0] if hasattr(model, "transform") else "(invalid extractive model)"

        # Save record
        record.update({
            "clean_text": cleaned_list[0],
            "predicted_topic": topic_out,
            "predicted_sentiment": sentiment_out,
            "summary": summary
        })
        save_to_store([record])

        # ----------------------------
        # Neatly formatted results
        # ----------------------------
        st.success("✅ Prediction complete and saved!")
        st.header("📊 Results")

        st.markdown(f"**🧩 Topic Model Used:** `{selected_topic_model}`")
        st.markdown(f"**💬 Sentiment Model Used:** `{selected_sentiment_model}`")
        st.markdown(f"**📝 Summarizer Used:** `{summarizer_choice}`")
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔮 Predicted Topic")
            if topic_out is not None:
                st.success(f"Topic ID: {topic_out}")
            else:
                st.info("No topic predicted.")

            st.subheader("😊 Sentiment")
            if sentiment_out:
                if sentiment_out.lower() == "positive":
                    st.success("Positive 🙂")
                elif sentiment_out.lower() == "negative":
                    st.error("Negative 🙁")
                else:
                    st.info(sentiment_out)
            else:
                st.warning("No sentiment predicted.")

        with col2:
            st.subheader("🧾 Summary")
            st.write(summary or "(No summary generated)")

        st.divider()
        st.subheader("🧹 Cleaned Text")
        st.text_area("Processed text", cleaned_list[0], height=150)

        st.subheader("📄 Original Text")
        with st.expander("Click to view full input text"):
            st.write(text)

    except Exception as e:
        st.error(f"Unexpected error: {e}")
