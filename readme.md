# 🕵️‍♂️Fake News Predictor

A Deep Learning application that detects fake news articles using a **LSTM** neural network. This project uses Natural Language Processing (NLP) to analyze linguistic patterns and determines if a news article is likely real or fake.

## 🚀 Features

* **Real-time Analysis:** Instant verification of news headlines and articles.
* **Deep Learning Model:** Powered by a custom trained LSTM model with Word2Vec embeddings.
* **Batch Processing:** Upload CSV files to analyze hundreds of articles at once.
* **Interactive Dashboard:** Built with [Streamlit](https://streamlit.io/) for a smooth user experience.
* **Visual Explanations:** Confidence scores and probability bars for every prediction.

## 🛠️ Tech Stack

* **Python 3.10+**
* **TensorFlow / Keras:** For the LSTM neural network.
* **Streamlit:** For the web interface.
* **NLTK:** For text preprocessing (tokenization, lemmatization).
* **Pandas & NumPy:** For data manipulation.

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/fake-news-detector.git](https://github.com/yourusername/fake-news-detector.git)
    cd fake-news-detector
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK data:**
    The app will automatically download necessary NLTK data (stopwords, wordnet) on the first run.

## 🏃‍♂️ Usage

1.  **Start the app:**
    ```bash
    streamlit run api.py
    ```

2.  **Open your browser:**
    The app should run automatically at `http://localhost:8501`.

3.  **Test it out:**
    * Paste a news article in the **"Analyze Single Article"** tab.
    * Or upload a CSV in the **"Batch Analysis"** tab.

## 📂 Project Structure

```text
├── api.py              # Main Streamlit Dashboard
├── predictor.py        # Inference Logic & Model Loading
├── requirements.txt    # Dependencies
├── .gitignore          # Ignored files
└── model/              # (Optional) Folder for model files
    ├── fake_news_lstm_model.keras
    └── tokenizer.pkl