import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.pre_processing.text_processing import preprocess_series
from src.topic_modelling.topic_utils import display_topics
import joblib
import os

# Load data
df = pd.read_csv("cleaned_dataset.csv")
texts = preprocess_series(df["text"])

# Vectorizer for LDA
count_vectorizer = CountVectorizer(max_features=10000, stop_words="english")
X_counts = count_vectorizer.fit_transform(texts)

# LDA Model
lda = LatentDirichletAllocation(
    n_components=20,
    random_state=42,
    learning_method="batch",
    max_iter=10
)
lda.fit(X_counts)

# Display topics
print("\n\ud83d\udd39 LDA Topics:\n")
display_topics(lda, count_vectorizer.get_feature_names_out())

# Save model + vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(lda, "models/lda_model.pkl")
joblib.dump(count_vectorizer, "models/lda_vectorizer.pkl")
print("\n\u2705 LDA model and vectorizer saved.")
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.pre_processing.text_processing import preprocess_series
from src.topic_modelling.topic_utils import display_topics
import joblib
import os

# Load data
df = pd.read_csv("cleaned_dataset.csv")
texts = preprocess_series(df["text"])

# Vectorizer for LDA
count_vectorizer = CountVectorizer(max_features=10000, stop_words="english")
X_counts = count_vectorizer.fit_transform(texts)

# LDA Model
lda = LatentDirichletAllocation(
    n_components=20,
    random_state=42,
    learning_method="batch",
    max_iter=10
)
lda.fit(X_counts)

# Display topics
print("\n🔹 LDA Topics:\n")
display_topics(lda, count_vectorizer.get_feature_names_out())

# Save model + vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(lda, "models/lda_model.pkl")
joblib.dump(count_vectorizer, "models/lda_vectorizer.pkl")
print("\n✅ LDA model and vectorizer saved.")
