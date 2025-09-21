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


 # Mapping from full category to single name
category_map = {
    'comp.sys.ibm.pc.hardware': 'hardware',
    'soc.religion.christian': 'religion',
    'sci.space': 'space',
    'alt.atheism': 'atheism',
    'comp.graphics': 'graphics',
    'comp.os.ms-windows.misc': 'windows',
    'comp.sys.mac.hardware': 'mac',
    'comp.windows.x': 'xwindows',
    'misc.forsale': 'forsale',
    'rec.autos': 'autos',
    'rec.motorcycles': 'motorcycles',
    'rec.sport.baseball': 'baseball',
    'rec.sport.hockey': 'hockey',
    'sci.crypt': 'crypt',
    'sci.electronics': 'electronics',
    'sci.med': 'med',
    'talk.politics.guns': 'guns',
    'talk.politics.mideast': 'mideast',
    'talk.politics.misc': 'politics',
    'talk.religion.misc': 'religionmisc',
}

for text in examples:
    prediction = model.predict([text])[0]
    short_pred = category_map.get(prediction, prediction.split(".")[-1])
    print(f"📝 Text: {text}")
    print(f"👉 Predicted Category: {short_pred}\n")