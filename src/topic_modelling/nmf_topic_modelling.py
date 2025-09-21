import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from src.pre_processing.text_processing import preprocess_series
from src.topic_modelling.topic_utils import display_topics
import joblib
import os

# Load data
df = pd.read_csv("cleaned_dataset.csv")
texts = preprocess_series(df["text"])

# Vectorizer for NMF
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# NMF Model
nmf = NMF(n_components=20, random_state=42, max_iter=200)
nmf.fit(X_tfidf)

# Display topics
print("\n\ud83d\udd39 NMF Topics:\n")
display_topics(nmf, tfidf_vectorizer.get_feature_names_out())

# Save model + vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(nmf, "models/nmf_model.pkl")
joblib.dump(tfidf_vectorizer, "models/nmf_vectorizer.pkl")
print("\n\u2705 NMF model and vectorizer saved.")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from src.pre_processing.text_processing import preprocess_series
from src.topic_modelling.topic_utils import display_topics
import joblib
import os

# Load data
df = pd.read_csv("cleaned_dataset.csv")
texts = preprocess_series(df["text"])

# Vectorizer for NMF
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# NMF Model
nmf = NMF(n_components=20, random_state=42, max_iter=200)
nmf.fit(X_tfidf)

# Display topics
print("\n🔹 NMF Topics:\n")
display_topics(nmf, tfidf_vectorizer.get_feature_names_out())

# Save model + vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(nmf, "models/nmf_model.pkl")
joblib.dump(tfidf_vectorizer, "models/nmf_vectorizer.pkl")
print("\n✅ NMF model and vectorizer saved.")
