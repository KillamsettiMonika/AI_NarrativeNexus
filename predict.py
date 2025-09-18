import joblib

# Load trained model
model_path = "models/text_classifier.pkl"
model = joblib.load(model_path)

# Test text samples
examples = [
    "The new Nvidia graphics card performs extremely well.",
    "The church gathering discussed the importance of prayer.",
    "The government announced new policies on space exploration."
]

for text in examples:
    prediction = model.predict([text])[0]
    print(f"📝 Text: {text}")
    print(f"👉 Predicted Category: {prediction}\n")