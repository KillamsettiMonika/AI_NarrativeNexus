import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Paths
train_path = "datasets/amazon_rev/amazon_reviews_train.csv"
model_path = "models/amazon_lstm.h5"
tokenizer_path = "models/amazon_lstm_tokenizer.pkl"

# Params
MAX_WORDS = 20000
MAX_LEN = 200
EPOCHS = 3
BATCH_SIZE = 128

# Load training data
print("📂 Loading training data...")
train_df = pd.read_csv(train_path, nrows=50000)
train_df["text"] = (train_df["title"].astype(str) + " " + train_df["content"].astype(str))

X_train, y_train = train_df["text"], train_df["label"]

# Tokenize
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)

# Build LSTM model
print("🧠 Building LSTM model...")
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train
print("🚀 Training LSTM...")
history = model.fit(
    X_train_seq, np.array(y_train),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_split=0.1
)

# Save model + tokenizer
os.makedirs("models", exist_ok=True)
model.save(model_path)
joblib.dump(tokenizer, tokenizer_path)

print(f"✅ Model saved to {model_path}")
print(f"✅ Tokenizer saved to {tokenizer_path}")
