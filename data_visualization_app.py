# data_visualization_app.py
import os
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# CONFIG
# ----------------------------
DATASETS_DIR = "datasets"
MODELS_DIR = "models"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DATA_STORE = os.path.join(DATA_DIR, "data_store.csv")

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title="📊 NarrativeNexus - Insights Dashboard", layout="wide")
st.title("📊 NarrativeNexus — Model & Dataset Visualization Dashboard")

st.markdown("""
Visualize performance and dataset insights for:
- 🧩 Topic Modeling (LDA / NMF)
- 💬 Sentiment Analysis (Random Forest / LSTM)
- 📝 Text Summarization (Abstractive / Extractive)
""")

# ----------------------------
# SIDEBAR: MODEL CATEGORY SELECTION
# ----------------------------
st.sidebar.header("⚙️ Configuration")

category = st.sidebar.selectbox(
    "Select Model Category",
    ["Topic Modeling", "Sentiment Analysis", "Text Summarization"]
)

# Category-based model filtering
if category == "Topic Modeling":
    model_keywords = ["lda", "nmf"]
elif category == "Sentiment Analysis":
    model_keywords = ["rf", "random_forest", "lstm", "classifier"]
elif category == "Text Summarization":
    model_keywords = ["abstractive", "extractive"]
else:
    model_keywords = []

# Get matching models
all_models = [f for f in os.listdir(MODELS_DIR) if f.endswith((".pkl", ".h5"))]
filtered_models = [m for m in all_models if any(k in m.lower() for k in model_keywords)]

if not filtered_models:
    st.warning("⚠️ No matching models found for this category.")
else:
    selected_model = st.sidebar.selectbox("Select Model", filtered_models)

# ----------------------------
# DATASET SELECTION
# ----------------------------
dataset_choices = [f for f in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, f))]
selected_dataset = st.sidebar.selectbox("Select Dataset Folder", dataset_choices)

# Try to find a data file
dataset_path = None
for root, dirs, files in os.walk(os.path.join(DATASETS_DIR, selected_dataset)):
    for file in files:
        if file.endswith((".csv", ".txt")):
            dataset_path = os.path.join(root, file)
            break

if not dataset_path:
    st.error("Dataset file not found inside selected folder.")
    st.stop()

# ----------------------------
# LOAD DATA
# ----------------------------
st.success(f"✅ Loaded dataset: {dataset_path}")
try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    st.error(f"Error reading dataset: {e}")
    st.stop()

st.dataframe(df.head())

# ----------------------------
# BASIC DATA SUMMARY
# ----------------------------
st.subheader("📄 Dataset Summary")
st.write(df.describe(include="all"))

# ----------------------------
# MODEL INFO SECTION
# ----------------------------
if filtered_models:
    st.subheader("🧠 Model Information")
    model_path = os.path.join(MODELS_DIR, selected_model)
    try:
        model = joblib.load(model_path)
        st.success(f"Model '{selected_model}' loaded successfully.")
    except Exception as e:
        st.error(f"Unable to load model: {e}")

# ----------------------------
# CATEGORY-SPECIFIC VISUALIZATIONS
# ----------------------------
if category == "Sentiment Analysis":
    if "predicted_sentiment" in df.columns:
        st.subheader("📊 Sentiment Distribution")
        sent_counts = df["predicted_sentiment"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=sent_counts.index, y=sent_counts.values, ax=ax, palette=["green", "red"])
        plt.title("Sentiment Analysis Results")
        st.pyplot(fig)
    else:
        st.info("No 'predicted_sentiment' column found in dataset.")

elif category == "Topic Modeling":
    if "predicted_topic" in df.columns:
        st.subheader("🧩 Topic Distribution")
        topic_counts = df["predicted_topic"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=topic_counts.index, y=topic_counts.values, ax=ax, palette="viridis")
        plt.title("Topic Model Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("No 'predicted_topic' column found in dataset.")

elif category == "Text Summarization":
    if "summary" in df.columns and "text" in df.columns:
        st.subheader("📝 Summary vs Original Text Lengths")
        df["original_len"] = df["text"].astype(str).apply(len)
        df["summary_len"] = df["summary"].astype(str).apply(len)
        fig, ax = plt.subplots()
        sns.histplot(df[["original_len", "summary_len"]], kde=True, multiple="dodge", ax=ax)
        plt.title("Original vs Summarized Text Lengths")
        st.pyplot(fig)
    else:
        st.info("No 'summary' or 'text' column found for summarization metrics.")

# ----------------------------
# CONFUSION MATRIX DISPLAY
# ----------------------------
confusion_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
if os.path.exists(confusion_path) and category == "Sentiment Analysis":
    st.subheader("📉 Model Performance (Confusion Matrix)")
    st.image(confusion_path, caption="Model Confusion Matrix", use_container_width=True)

# ----------------------------
# ALL MODELS TABLE
# ----------------------------
st.subheader("📦 Available Models")
st.write(pd.DataFrame(all_models, columns=["Model Files"]))
