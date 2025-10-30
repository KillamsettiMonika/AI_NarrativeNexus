# text_summarization/extractive_summarizer.py

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pickle

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load BBC dataset
data_path = "datasets/BBC articles/bbc-text.csv"  # adjust to your actual CSV path
df = pd.read_csv(data_path)
texts = df["text"].astype(str).tolist()

# Preprocess
def clean_text(text):
    return " ".join(text.split())

# Build TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform([clean_text(t) for t in texts])

# Build similarity graph
similarity_matrix = cosine_similarity(X)
nx_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(nx_graph)

# Select top 5 sentences per document
summaries = []
for i, text in enumerate(texts):
    sentences = text.split(". ")
    ranked = sorted(((scores.get(j, 0), s) for j, s in enumerate(sentences)), reverse=True)
    summary = ". ".join([s for _, s in ranked[:3]])
    summaries.append(summary)

# Save summaries
summary_df = pd.DataFrame({"original_text": texts, "summary": summaries})
summary_df.to_csv("text_summarization/extractive_summaries.csv", index=False)
print("✅ Extractive summaries saved to text_summarization/extractive_summaries.csv")

# Save model artifacts
with open("models/extractive_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Extractive summarization model saved in models/ folder")
