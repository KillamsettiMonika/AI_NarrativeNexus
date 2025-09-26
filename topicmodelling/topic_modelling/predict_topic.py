import joblib
from src.pre_processing.text_processing import preprocess_series
import numpy as np

def get_topic_keywords(model, vectorizer, n_top_words=5):
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_keywords.append(", ".join(top_features))
    return topic_keywords

# Load models
lda = joblib.load("models/lda_model.pkl")
count_vectorizer = joblib.load("models/lda_vectorizer.pkl")

nmf = joblib.load("models/nmf_model.pkl")
tfidf_vectorizer = joblib.load("models/nmf_vectorizer.pkl")

examples = [
    "The new graphics card from NVIDIA has amazing performance for 3D rendering.",
    "God does not exist, and religion is just a human creation.",
    "NASA discovered water on Mars, confirming planetary research findings."
]

examples_clean = preprocess_series(examples)

# Get topic keywords
lda_topic_keywords = get_topic_keywords(lda, count_vectorizer)
nmf_topic_keywords = get_topic_keywords(nmf, tfidf_vectorizer)

# ----- LDA -----
lda_features = count_vectorizer.transform(examples_clean)
lda_topics = lda.transform(lda_features)
print("\n🔹 LDA Predictions:")
for text, topic in zip(examples, np.argmax(lda_topics, axis=1)):
    print(f"\nText: {text}\nPredicted Topic: {topic} ({lda_topic_keywords[topic]})")

# ----- NMF -----
nmf_features = tfidf_vectorizer.transform(examples_clean)
nmf_topics = nmf.transform(nmf_features)
print("\n🔹 NMF Predictions:")
for text, topic in zip(examples, np.argmax(nmf_topics, axis=1)):
    print(f"\nText: {text}\nPredicted Topic: {topic} ({nmf_topic_keywords[topic]})")
