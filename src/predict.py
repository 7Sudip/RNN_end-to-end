from tensorflow.keras.models import load_model
from preprocess import preprocess_text,text_to_sequence
from config import MODEL_PATH 

# Load model 
model = load_model(MODEL_PATH)


def predict_sentiment(text):
    sequence = text_to_sequence(preprocess_text(text))
    prediction = model.predict(sequence)[0][0]

    if prediction >= 0.5:
        return "Positive 😊", float(prediction)
    else:
        return "Negative 😠", float(prediction)
