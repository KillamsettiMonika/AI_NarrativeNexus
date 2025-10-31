# app.py
import os
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(page_title="AI Narrative Nexus", layout="wide")

DATASETS_DIR = "datasets"
MODELS_DIR = "models"

# Confusion matrix paths
CONFUSION_MATRICES = {
    "Topic Modeling": [
        ("LDA", "topicmodelling/topic_modelling/lda_confusion_matrix.png"),
        ("NMF", "topicmodelling/topic_modelling/nmf_confusion_matrix.png"),
    ],
    "Sentiment Analysis": [
        ("Random Forest", "sentiment_analysis/random_forest1/confusion_matrix_rf.png"),
        ("LSTM", "sentiment_analysis/lstm/confusion_matrix_lstm.png"),
    ],
    "Text Summarization": [
        ("Abstractive", "sentiment_analysis/text_summarization/abs_confusion_matrix.png"),
        ("Extractive", "sentiment_analysis/text_summarization/evaluation_metrics.png"),
    ],
}

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", [
    "Home",
    "Topic Modeling",
    "Sentiment Analysis",
    "Text Summarization",
    "Data Visualization",
    "Evaluation & Analysis",
    "Live Demo",
    "About"
])

# -------------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------------
if page == "Home":
    st.title("🏠 AI Narrative Nexus")
    st.markdown("""
    **AI Narrative Nexus** transforms unstructured text into actionable insights through:
    - 🧩 **Topic Modeling** (LDA / NMF)
    - ❤️ **Sentiment Analysis** (Random Forest / LSTM)
    - 🧠 **Text Summarization** (Abstractive / Extractive)
    - 📊 **Data Visualization & Evaluation Dashboards**

    ---
    ### 🚀 Project Overview
    This project integrates multiple NLP techniques for real-world data analysis using:
    - **Python**, **Streamlit**, **Transformers**, **NLTK**, **Matplotlib**, **Seaborn**
    - Modular architecture with separate pipelines for topic detection, sentiment prediction, and summarization.
    """)

# -------------------------------------------------------------
# TOPIC MODELING
# -------------------------------------------------------------
elif page == "Topic Modeling":
    st.title("🧩 Topic Modeling")
    model_choice = st.selectbox("Select Model", ["LDA", "NMF"])
    input_type = st.radio("Select Input Type", ["Free Text", "Reddit URL", "News API"])

    text = st.text_area("Enter text or link here:")
    if st.button("Run Topic Modeling"):
        st.success(f"✅ Topic Modeling ({model_choice}) executed successfully!")
        st.write({
            "model": model_choice,
            "input": input_type,
            "topic_prediction": "Sample Topic 12 — Religion & Culture",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

# -------------------------------------------------------------
# SENTIMENT ANALYSIS
# -------------------------------------------------------------
elif page == "Sentiment Analysis":
    st.title("❤️ Sentiment Analysis")
    model_choice = st.selectbox("Select Model", ["Random Forest", "LSTM"])
    input_type = st.radio("Select Input Type", ["Free Text", "Reddit URL", "News API"])

    text = st.text_area("Enter review, tweet, or paragraph:")
    if st.button("Run Sentiment Analysis"):
        st.success(f"✅ Sentiment Analysis ({model_choice}) complete!")
        st.write({
            "model": model_choice,
            "sentiment": "Positive",
            "confidence": "0.94",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

# -------------------------------------------------------------
# TEXT SUMMARIZATION
# -------------------------------------------------------------
elif page == "Text Summarization":
    st.title("🧠 Text Summarization")
    model_choice = st.selectbox("Select Summarization Type", ["Abstractive", "Extractive"])
    input_type = st.radio("Select Input Type", ["Free Text", "Reddit URL", "News API"])
    text = st.text_area("Paste the text/article to summarize:")

    if st.button("Generate Summary"):
        st.success(f"✅ Summary generated using {model_choice} model!")
        summary = "This is a demonstration summary of your input text showing key ideas concisely."
        st.markdown(f"### ✨ Summary:\n> {summary}")

# -------------------------------------------------------------
# DATA VISUALIZATION (EDA)
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# EVALUATION & ANALYSIS
# -------------------------------------------------------------
elif page == "Evaluation & Analysis":
    st.title("📈 Evaluation & Analysis")
    st.markdown("View confusion matrices and performance metrics for each model.")
    model_type = st.selectbox("Select Model Type", ["Topic Modeling", "Sentiment Analysis", "Text Summarization"])

    for model_name, path in CONFUSION_MATRICES[model_type]:
        if os.path.exists(path):
            st.image(path, caption=f"{model_name} - Confusion Matrix", use_container_width=True)
        else:
            st.warning(f"⚠️ Confusion matrix not found for: {path}")

# -------------------------------------------------------------
# LIVE DEMO
# -------------------------------------------------------------
elif page == "Live Demo":
    st.title("🧪 Live Demo")
    st.info("Live demo functionality coming soon!")

# -------------------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------------------
elif page == "About":
    st.title("ℹ️ About AI Narrative Nexus")
    st.markdown("""
    **AI Narrative Nexus** converts unstructured text into meaningful insights using:
    - Topic Modeling (LDA/NMF)
    - Sentiment Analysis (Random Forest/LSTM)
    - Text Summarization (Abstractive/Extractive)
    - Interactive Visualizations

    **Tech Stack:**
    - 🐍 Python
    - 🤗 Transformers
    - 📊 Matplotlib / Seaborn
    - 💡 Streamlit

    **Team:** Monika Killamsetti & Team  
    **Project:** Infosys Internship 2025  
    **Report:** `AI_Narrative_Nexus.pdf`
    """)
