import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pickle
import evaluate
import matplotlib.pyplot as plt

# -----------------------------
# 1. Setup folders
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("text_summarization", exist_ok=True)

# -----------------------------
# 2. Load dataset
# -----------------------------
data_path = "datasets/BBC articles/bbc-text.csv"  # adjust if path differs
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

print("📥 Loading dataset...")
df = pd.read_csv(data_path)
texts = df["text"].astype(str).tolist()

# -----------------------------
# 3. Preprocessing
# -----------------------------
def clean_text(text):
    return " ".join(text.split())

print("🧹 Cleaning and vectorizing text...")
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform([clean_text(t) for t in texts])

# -----------------------------
# 4. Similarity + Graph + PageRank
# -----------------------------
print("🔗 Building similarity graph...")
similarity_matrix = cosine_similarity(X)
nx_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(nx_graph)

# -----------------------------
# 5. Generate extractive summaries
# -----------------------------
print("🧠 Generating extractive summaries...")
summaries = []
for i, text in enumerate(texts):
    sentences = text.split(". ")
    ranked = sorted(((scores.get(j, 0), s) for j, s in enumerate(sentences)), reverse=True)
    summary = ". ".join([s for _, s in ranked[:3]])
    summaries.append(summary)

# Save summaries to CSV
summary_path = "text_summarization/extractive_summaries.csv"
pd.DataFrame({"original_text": texts, "summary": summaries}).to_csv(summary_path, index=False, encoding="utf-8")
print(f"✅ Extractive summaries saved to: {summary_path}")

# Save model vectorizer
model_path = "models/extractive_vectorizer.pkl"
with open(model_path, "wb") as f:
    pickle.dump(vectorizer, f)
print(f"✅ Vectorizer saved to: {model_path}")

# -----------------------------
# 6. Evaluate using ROUGE metrics
# -----------------------------
print("\n📊 Evaluating extractive summaries using ROUGE metrics...")

try:
    rouge = evaluate.load("rouge")

    # Create pseudo-references: first few sentences of each text
    references = [t.split(". ")[0] for t in texts[:10]]
    preds = summaries[:10]

    results = rouge.compute(predictions=preds, references=references)
    print("📈 ROUGE Evaluation Results:")
    for metric, score in results.items():
        print(f"   {metric.upper():<12}: {score:.4f}")

    # -----------------------------
    # 7. Plot and save as image
    # -----------------------------
    print("🖼️ Generating evaluation metrics plot...")
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=["#4CAF50", "#2196F3", "#FFC107", "#E91E63"])
    plt.title("ROUGE Evaluation Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    img_path = "text_summarization/evaluation_metrics.png"
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"✅ Evaluation metrics image saved to: {img_path}")

except Exception as e:
    print(f"⚠️ ROUGE evaluation skipped: {e}")
