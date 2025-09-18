# text_processing.py
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab") 
# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def nlp_preprocess(text):
    """Clean, tokenize, remove stopwords, and lemmatize text."""
    if not isinstance(text, str):
        return ""

    # Tokenize and lowercase
    tokens = nltk.word_tokenize(text.lower())

    # Keep only alphabetic tokens, remove stopwords
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

def preprocess_series(X):
    """Apply preprocessing to a Pandas Series of texts."""
    return [nlp_preprocess(t) for t in X]

print("✅ Text preprocessing functions ready.")
