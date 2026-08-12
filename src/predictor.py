import os
import re
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILENAME = "fake_news_bilstm_model_11aug.keras"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

DEFAULT_TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pickle")
DEFAULT_CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
print('Tokenizer path', DEFAULT_TOKENIZER_PATH)

# Fallback only used if model_config.json (saved by the training notebook) is missing.
FALLBACK_MAX_LEN = 300


def _load_max_len(config_path: str = DEFAULT_CONFIG_PATH, fallback: int = FALLBACK_MAX_LEN) -> int:
    """Load the MAX_LEN the model was actually trained with.

    The training notebook saves this to /kaggle/working/model_config.json.
    Download that file alongside the model + tokenizer into src/model/ —
    hardcoding a guessed MAX_LEN is what was causing the mismatch.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            max_len = int(cfg["MAX_LEN"])
            print(f"Loaded MAX_LEN={max_len} from {config_path}")
            return max_len
        except Exception as e:
            print(f"WARNING: could not read MAX_LEN from {config_path} ({e}); "
                  f"falling back to {fallback}. This may cause a shape mismatch "
                  f"if it doesn't match training.")
    else:
        print(f"WARNING: {config_path} not found; falling back to MAX_LEN={fallback}. "
              f"Download model_config.json from the training notebook's output to fix this properly.")
    return fallback


# Same stopword list NLTK's stopwords.words('english') returns — hardcoded so the
# deployed app doesn't need an nltk.download() call (and matching internet access)
# at runtime, and so it can NEVER silently drift from what training used.
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
    'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
    "wouldn't",
}


def clean_text(text: str) -> str:
    """Must exactly mirror the cleaning used in the training notebook, or the
    model sees a different 'flavor' of text than it learned from."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)   # URLs
    text = re.sub(r'<.*?>', ' ', text)                     # HTML tags
    text = re.sub(r'\S+@\S+', ' ', text)                   # emails
    text = re.sub(r'[^a-z\s]', ' ', text)                  # letters only (matches training: no digits kept)
    text = re.sub(r'\s+', ' ', text).strip()               # collapse whitespace
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)


class TokenizerPredictor:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        max_len: int = None,
    ):
        self.max_len = max_len if max_len is not None else _load_max_len(config_path)

        # 1. MODEL
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model file not found at {model_path}. "
                f"Make sure {MODEL_FILENAME} is in src/model "
                "and committed to the repo."
            )

        print(f"Loading model from: {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")

        # 2. TOKENIZER
        try:
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            print("Tokenizer loaded successfully:", type(self.tokenizer))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Tokenizer from {tokenizer_path}: {e}"
            )

    def _compose(self, title, text):
        title = title or ""
        text = text or ""
        return (str(title) + " " + str(text)).strip()

    def _preprocess_texts(self, texts):
        cleaned = [clean_text(t) for t in texts]

        if not hasattr(self, "tokenizer") or not hasattr(self.tokenizer, "texts_to_sequences"):
            raise AttributeError("Tokenizer not loaded properly.")

        seqs = self.tokenizer.texts_to_sequences(cleaned)
        # NOTE: must match training (padding='post', truncating='post').
        # Using 'pre' here (as before) fed the model padding in a position
        # it never saw during training, which hurts accuracy silently.
        padded = pad_sequences(
            seqs, maxlen=self.max_len, padding="post", truncating="post"
        )
        return padded

    def predict_single_news(self, title: str, text: str):
        combined = self._compose(title, text)

        if not combined:
            return {"label": 0, "label_str": "uncertain", "probability": 0.0}

        x = self._preprocess_texts([combined])

        probs = self.model.predict(x, verbose=0).reshape(-1)
        p = float(probs[0])

        # IMPORTANT: verify this mapping is correct for your trained model.
        # The dataset's documentation is inconsistent about whether 0 or 1
        # means "fake" -- test with the app's built-in Real:/Fake: samples
        # and flip this ternary if predictions come out backwards.
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