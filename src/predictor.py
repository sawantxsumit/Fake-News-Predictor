import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests 
import time
import gdown  
# --- MODEL DOWNLOAD CONFIGURATION ---
# MODEL_FILE_ID = "1tnLr7KNZAU3sGYVugWIS8A3wFZqcXIQs"
# TOKENIZER_FILE_ID = "10egovgFOO-XffG9ML4crgwLqOmV6wITM"

# # Base directory of this file 
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_DIR = os.path.join(BASE_DIR, "model_cache")
# os.makedirs(MODEL_DIR, exist_ok=True)

# MODEL_FILENAME = "fake_news_lstm_model.keras"
# DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# # Local paths where files will be saved during deployment
# DEFAULT_TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
# MAX_LEN = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_cache")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILENAME = "fake_news_lstm_model.keras"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

DEFAULT_TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
MAX_LEN = 300


def clean_text(text: str) -> str:
    """Basic text cleaning used before tokenization."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def download_file_from_gdrive(file_id: str, destination: str, min_size_bytes: int = 1024 * 1024):
    """
    Download a file from Google Drive using gdown.

    - Skips download if file already exists and is larger than `min_size_bytes`
      (to avoid re-downloading on every run).
    - If file exists but is too small (likely an HTML error page), it re-downloads.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    if os.path.exists(destination):
        size = os.path.getsize(destination)
        if size >= min_size_bytes:
            print(f"File {os.path.basename(destination)} already exists "
                  f"({size/1024/1024:.2f} MB). Skipping download.")
            return
        else:
            print(f"File {os.path.basename(destination)} exists but is too small "
                  f"({size} bytes). Re-downloading...")

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {os.path.basename(destination)} from Google Drive...")
    gdown.download(url, destination, quiet=False)

    # Simple sanity check
    final_size = os.path.getsize(destination)
    if final_size < min_size_bytes:
        raise RuntimeError(
            f"Downloaded file {os.path.basename(destination)} is too small "
            f"({final_size} bytes). It may not be a valid model/tokenizer file."
        )
    print(f"Successfully downloaded {os.path.basename(destination)} "
          f"({final_size/1024/1024:.2f} MB).")


class TokenizerPredictor:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        max_len: int = MAX_LEN,
    ):
        def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
                 max_len: int = MAX_LEN):

            self.max_len = max_len

            # 1. MODEL: just load from repo; no download
            if not os.path.exists(model_path):
                raise RuntimeError(
                    f"Model file not found at {model_path}. "
                    "Make sure fake_news_lstm_model.keras is in src/model_cache "
                    "and committed to the repo."
                )

            print(f"Loading model from: {model_path}")
            try:
                self.model = tf.keras.models.load_model(model_path)
                print("Model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load Keras model: {e}")

            # 2. TOKENIZER: either load from disk, or still download from Drive if you like
            try:
                with open(tokenizer_path, "rb") as f:
                    self.tokenizer = pickle.load(f)
                print("Tokenizer loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load Tokenizer from {tokenizer_path}: {e}")

    def _compose(self, title, text):
        title = title or ""
        text = text or ""
        return (str(title) + " " + str(text)).strip()

    def _preprocess_texts(self, texts):
        cleaned = [clean_text(t) for t in texts]
        if not hasattr(self, "tokenizer"):
            raise AttributeError("Tokenizer not loaded properly.")

        seqs = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(seqs, maxlen=self.max_len, padding="pre", truncating="pre")
        return padded

    def predict_single_news(self, title: str, text: str):
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
            "probability": p,
        }

    def predict_batch(self, texts, batch_size=512):
        results = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch_texts = texts[i: i + batch_size]
            x = self._preprocess_texts(batch_texts)
            probs = self.model.predict(x, verbose=0).reshape(-1)

            for p in probs:
                lab = 1 if float(p) > 0.5 else 0
                results.append(
                    {
                        "label": int(lab),
                        "label_str": "real" if lab == 1 else "fake",
                        "probability": float(p),
                    }
                )

        return results
