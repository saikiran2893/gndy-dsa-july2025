from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import re


# Initialize FastAPI app
app = FastAPI()


# Load model and tokenizer
MODEL_PATH = "sentiment_model_2.keras"
TOKENIZER_PATH = "tokenizer_2.joblib"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = load(TOKENIZER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

# Define request and response models
class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    input_text:str
    cleaned_text:str
    sentiment: str
    confidence: float
    probabilities: dict

def preprocess_text(string):
    removelist = r"!*,."
    result = re.sub('<.*?>','',string)          #remove HTML tags
    result = re.sub('https://.*','',result)   #remove URLs
    result = re.sub(f"[{removelist}]", "", result)  #remove non-alphanumeric characters
    result = result.lower()
    return result

# Prediction function
def predict_sentiment(text):
    # Preprocess the input text
    cleaned_text = preprocess_text(text)
    print(f'The cleaned text is {cleaned_text}')
    # Tokenize and pad the text
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded, verbose=0)
    print(prediction[0])
    sentiment_labels = ['negative', 'neutral', 'positive']
    print(prediction.argmax())
    predicted_sentiment = sentiment_labels[prediction.argmax()]
    confidence = float(prediction.max())

    return {
        'input_text':text,
        'cleaned_text':cleaned_text,
        'sentiment': predicted_sentiment,
        'confidence': confidence,
        'probabilities': {
            label: float(prob)
            for label, prob in zip(sentiment_labels, prediction[0])
        }
    }

# API endpoint
@app.post("/predict", response_model=SentimentResponse)
def get_sentiment(input: TextInput):
    try:
        result = predict_sentiment(input.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)