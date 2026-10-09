import streamlit as st
import pickle
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from samples import SAMPLES

# Directory containing app.py
BASE_DIR = Path(__file__).resolve().parent

# Model directory
MODEL_DIR = BASE_DIR / "model"

# Model files
MODEL_PATH = MODEL_DIR / "ISOT_model_27aug.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pickle"


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Fake News Predictor",
    page_icon="🕵️",
    layout="wide"
)


# LOAD MODEL AND TOKENIZER

@st.cache_resource
def load_resources():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at: {TOKENIZER_PATH}"
        )

    # Load model
    model = load_model(MODEL_PATH)

    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_resources()


# SIDEBAR


# ==========================================
# SIDEBAR - FAKE NEWS PREDICTOR
# ==========================================


with st.sidebar:
    st.title("🕵️ Fake News Predictor")

    st.caption("Deep Learning • Natural Language Processing")

    st.markdown("---")

    st.subheader("About the Project")

    st.markdown(
        """
        An end-to-end NLP application that uses a
        **Bidirectional LSTM (Bi-LSTM)** neural network
        to classify news articles as **REAL or FAKE**.

        The model learns linguistic and sequential patterns
        from labeled articles in the ISOT Fake News Dataset.
        """
    )

    st.markdown("---")

    st.subheader("How It Works")

    with st.expander("1. Dataset Preparation", expanded=False):
        st.markdown(
            """
            - Load real and fake news datasets.
            - Remove missing records and duplicates.
            - Assign binary labels:
                - `0` — Fake
                - `1` — Real
            - Combine the headline and article body.
            """
        )

    with st.expander("2. Text Preprocessing", expanded=False):
        st.markdown(
            """
            - Convert text to lowercase.
            - Normalize whitespace.
            - Tokenize using the Keras Tokenizer.
            - Limit the vocabulary to 30,000 words.
            - Handle unseen words using `<OOV>`.
            - Pad or truncate sequences to 750 tokens.
            """
        )

    with st.expander("3. Bi-LSTM Architecture", expanded=False):
        st.markdown(
            """
            **Network architecture**

            1. Embedding — 128 dimensions
            2. Bidirectional LSTM — 64 units per direction
            3. Dense — 64 neurons, ReLU
            4. Dropout — 0.50
            5. Output — 1 sigmoid neuron

            The network learns contextual patterns from
            sequences of words to perform binary classification.
            """
        )

    with st.expander("4. Training Methodology", expanded=False):
        st.markdown(
            """
            - Optimizer: Adam
            - Loss: Binary Cross-Entropy
            - Batch size: 64
            - Maximum epochs: 15
            - EarlyStopping: patience of 3 epochs
            - ReduceLROnPlateau: halves the learning rate
              when validation loss stops improving
            - ModelCheckpoint: saves the best validation-loss model
            """
        )

    with st.expander("5. Evaluation Metrics", expanded=False):
        st.markdown(
            """
            The notebook evaluates the model using:

            - Accuracy
            - Precision
            - Recall
            - F1-score
            - Binary cross-entropy loss
            - Confusion matrix

            Training and validation curves are also plotted
            to inspect learning behaviour.
            """
        )

    st.markdown("---")

    st.subheader("Technology Stack")

    st.markdown(
        """
        - **Language:** Python
        - **Data Processing:** Pandas, NumPy
        - **Deep Learning:** TensorFlow / Keras
        - **Evaluation:** Scikit-learn
        - **Visualization:** Matplotlib, Seaborn
        - **Interface:** Streamlit
        """
    )

    st.markdown("---")

    st.subheader("Model Information")

    st.markdown(
        """
        - Dataset: ISOT Fake News Dataset
        - Task: Binary text classification
        - Vocabulary limit: 30,000 words
        - Sequence length: 750 tokens
        - Classification threshold: 0.50
        """
    )

    st.info(
        "The prediction made by this model indicared the patterns learned by it during the training from ISOT fake news dataset"
        "It does not perfectly verify whether a news is factual."
    )


# MAIN TITLE

st.title("🕵️ Fake News Predictor")

st.write(
    "Enter a news article to check "
    "whether it is REAL or FAKE."
)

st.markdown("---")


if "headline" not in st.session_state:
    st.session_state.headline = ""

if "article_content" not in st.session_state:
    st.session_state.article_content = ""

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "Select a sample..."

# SAMPLE SELECTION

sample_options = [
    "Select a sample..."
] + list(SAMPLES.keys())


def load_sample():

    selected = st.session_state.sample_selector

    if selected != "Select a sample...":

        sample = SAMPLES[selected]

        st.session_state.headline = sample["headline"]
        st.session_state.article_content = sample["content"]

        st.session_state.selected_sample = selected


st.selectbox(
    "Select an article from the dataset",
    sample_options,
    key="sample_selector",
    on_change=load_sample
)


# INPUT SECTION

col1, col2 = st.columns(2)

# LEFT COLUMN

with col1:

    st.subheader("Input Details")

    headline = st.text_input(
        "Headline",
        key="headline",
        placeholder="Enter the news headline..."
    )

    article_content = st.text_area(
        "Article Content",
        key="article_content",
        height=350,
        placeholder="Paste the full news article here..."
    )

    analyze = st.button(
        "🔍 Analyze News Article",
        use_container_width=True
    )

# RIGHT COLUMN

with col2:

    st.subheader("Analysis Results")

    if analyze:

        if not article_content.strip():

            st.warning(
                "Please enter an article before analyzing."
            )

        else:

            # TOKENIZATION

            sequence = tokenizer.texts_to_sequences(
                [article_content]
            )

            # PADDING

            padded = pad_sequences(
                sequence,
                maxlen=750,
                padding="post",
                truncating="post"
            )

            # PREDICTION

            probability = model.predict(
                padded,
                verbose=0
            )[0][0]

            prediction = int(
                probability >= 0.5
            )

            # RESULT

            if prediction == 1:

                st.success("## ✅ REAL NEWS")

                st.metric(
                    "Probability of REAL",
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )

            else:

                st.error("## 🚨 FAKE NEWS")

                fake_probability = 1 - probability

                st.metric(
                    "Probability of FAKE",
                    f"{fake_probability * 100:.2f}%"
                )

                st.progress(
                    float(fake_probability)
                )

            # PREDICTION DETAILS

            st.markdown("---")

            st.write("### Prediction Details")

            st.write(
                f"**Model probability (REAL):** "
                f"{probability:.4f}"
            )

            st.write(
                "**Classification threshold:** 0.50"
            )


            selected = st.session_state.selected_sample

            if selected != "Select a sample...":

                actual_label = SAMPLES[
                    selected
                ]["actual_label"]

                actual_label_text = (
                    "REAL"
                    if actual_label == 1
                    else "FAKE"
                )

                predicted_label_text = (
                    "REAL"
                    if prediction == 1
                    else "FAKE"
                )

                
    else:

        st.info(
            "Select a sample article or enter your "
            "own article and click **Analyze Authenticity**."
        )