import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import time

# --- MODEL DOWNLOAD CONFIGURATION ---
# Replace these with your actual Google Drive File IDs
MODEL_FILE_ID = "1tnLr7KNZAU3sGYVugWIS8A3wFZqcXIQs"     
TOKENIZER_FILE_ID = "10egovgFOO-XffG9ML4crgwLqOmV6wITM"  

import os

# Base directory of this file (predictor.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model_cache folder next to predictor.py
MODEL_DIR = os.path.join(BASE_DIR, "model_cache")

MODEL_FILENAME = "fake_news_lstm_model.keras"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


# Local paths where files will be saved during deployment
# MODEL_DIR = "model_cache"
# DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_lstm_model.keras")
DEFAULT_TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
MAX_LEN = 300 

def clean_text(text: str) -> str:
    # ... (Keep your clean_text function the same)
    if text is None:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def download_file_from_gdrive(file_id, destination):
    
    
    # URL to access the file content
    URL = "https://docs.google.com/uc?export=download"
    
    # Check if the file already exists
    if os.path.exists(destination):
        print(f"File {os.path.basename(destination)} already exists. Skipping download.")
        return

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"File {os.path.basename(destination)} not found. Downloading from Google Drive...")
    
    session = requests.Session()
    
    # Initial request to get the download confirmation token
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check if a confirmation token is needed (for files too large to scan for viruses)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    # If token exists, make the second request with the token
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Raise an error if the download failed for any other reason
    response.raise_for_status()

    # Write the content to the file
    chunk_size = 32768
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  
                f.write(chunk)
    print(f"Successfully downloaded {os.path.basename(destination)}.")


class TokenizerPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, tokenizer_path: str = DEFAULT_TOKENIZER_PATH, max_len=MAX_LEN):
        self.max_len = max_len
        
        download_file_from_gdrive(MODEL_FILE_ID, model_path)
        download_file_from_gdrive(TOKENIZER_FILE_ID, tokenizer_path)

        print(f"Loading model from: {model_path}")
        print(model_path)
        try:
            self.model = tf.keras.models.load_model(model_path)
            print('Model loaded successfully')
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")

        try:
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        except Exception as e:
             raise RuntimeError(f"Failed to load Tokenizer: {e}")
        
    def _compose(self, title, text):
        title = title or ""
        text = text or ""
        return (str(title) + " " + str(text)).strip()
    
    def _preprocess_texts(self, texts):
        cleaned = [clean_text(t) for t in texts]
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("Tokenizer not loaded properly.")
            
        seqs = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(seqs, maxlen=self.max_len, padding='pre', truncating='pre')
        return padded
    
    def predict_single(self, title: str, text: str):
        combined = self._compose(title, text)

        if not combined:
            return {"label": 0, "label_str": "uncertain", "probability": 0.0}

        x = self._preprocess_texts([combined])
        
        probs = self.model.predict(x, verbose=0).reshape(-1)
        p = float(probs[0])
        
        label = 1 if p > 0.5 else 0
        label_str = "real" if label == 1 else "fake"
        
        return {
            "label": int(label), 
            "label_str": label_str, 
            "probability": p
        }
    
    def predict_batch(self, texts, batch_size=512):
        results = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            x = self._preprocess_texts(batch_texts)
            probs = self.model.predict(x, verbose=0).reshape(-1)
            
            for p in probs:
                lab = 1 if float(p) > 0.5 else 0
                results.append({
                    "label": int(lab), 
                    "label_str": "real" if lab == 1 else "fake", 
                    "probability": float(p)
                })
        
        return results