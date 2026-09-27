import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fake News Predictor",
    page_icon="📰",
    layout="wide"
)


# =========================================================
# DATASET PATHS
# =========================================================

FAKE_DATASET_PATH = "D:/Fake-News-Predictor/data/Fake.csv"
REAL_DATASET_PATH = "D:/Fake-News-Predictor/data/True.csv"


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_sample_dataset():

    # Load Fake News
    fake_df = pd.read_csv(FAKE_DATASET_PATH)

    # Load Real News
    real_df = pd.read_csv(REAL_DATASET_PATH)

    # Add labels
    fake_df["label"] = 0
    real_df["label"] = 1

    # Combine datasets
    df = pd.concat(
        [fake_df, real_df],
        ignore_index=True
    )

    # Remove rows where title/text is missing
    df = df.dropna(
        subset=["title", "text"]
    )

    # Remove empty articles
    df = df[
        (df["title"].str.strip() != "") &
        (df["text"].str.strip() != "")
    ]

    return df


# Load dataset
dataset = load_sample_dataset()


# =========================================================
# CREATE SAMPLE ARTICLES
# =========================================================

@st.cache_data
def create_samples(df):

    # Take 4 real articles
    real_samples = (
        df[df["label"] == 1]
        .sample(n=4, random_state=42)
    )

    # Take 4 fake articles
    fake_samples = (
        df[df["label"] == 0]
        .sample(n=4, random_state=42)
    )

    # Combine
    samples = pd.concat(
        [real_samples, fake_samples]
    )

    return samples


samples_df = create_samples(dataset)


# =========================================================
# CREATE SAMPLE OPTIONS
# =========================================================

sample_options = [
    "Select a sample..."
]

sample_mapping = {}


for index, row in samples_df.iterrows():

    actual_label = "Real" if row["label"] == 1 else "Fake"

    # Create a short title for dropdown
    title = str(row["title"]).strip()

    if len(title) > 90:
        title = title[:90] + "..."

    option = f"{actual_label}: {title}"

    sample_options.append(option)

    sample_mapping[option] = row


# =========================================================
# LOAD MODEL AND TOKENIZER
# =========================================================

@st.cache_resource
def load_resources():

    model = load_model(
        "D:/Fake-News-Predictor/src/model/ISOT_model_27aug.keras"
    )

    with open(
        "D:/Fake-News-Predictor/src/model/tokenizer.pickle",
        "rb"
    ) as f:

        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_resources()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📰 Fake News")

    st.markdown("### About")

    st.write(
        "A machine learning application that uses an LSTM model "
        "to classify news articles as REAL or FAKE."
    )

    st.markdown("---")

    st.markdown("### Dataset")

    st.write(
        f"**Total articles:** {len(dataset):,}"
    )

    st.write(
        f"**Real articles:** "
        f"{(dataset['label'] == 1).sum():,}"
    )

    st.write(
        f"**Fake articles:** "
        f"{(dataset['label'] == 0).sum():,}"
    )

    st.markdown("---")

    st.markdown("### How it works")

    st.write("**1. Input**")
    st.write("Enter or select a news article.")

    st.write("**2. Tokenization**")
    st.write("The article is converted into numerical sequences.")

    st.write("**3. LSTM Model**")
    st.write("The trained model analyzes the text.")

    st.write("**4. Prediction**")
    st.write("The model returns a REAL or FAKE classification.")


# =========================================================
# MAIN TITLE
# =========================================================

st.title("📰 Fake News Predictor")

st.write(
    "Enter a news article or select an article directly "
    "from the training dataset to test the prediction model."
)

st.markdown("---")


# =========================================================
# SAMPLE ARTICLE SELECTION
# =========================================================

selected_sample = st.selectbox(
    "Quick Load Sample News",
    sample_options
)


# =========================================================
# LOAD SELECTED SAMPLE
# =========================================================

if selected_sample != "Select a sample...":

    sample = sample_mapping[selected_sample]

    default_headline = str(sample["title"])
    default_content = str(sample["text"])

else:

    default_headline = ""
    default_content = ""


# =========================================================
# INPUT SECTION
# =========================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("Input Details")

    headline = st.text_input(
        "Headline (Optional)",
        value=default_headline,
        placeholder="e.g. Breaking News..."
    )

    article_content = st.text_area(
        "Article Content",
        value=default_content,
        height=300,
        placeholder="Paste the full news article here..."
    )

    analyze = st.button(
        "🔍 Analyze Authenticity",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

with col2:

    st.subheader("Analysis Results")

    if analyze:

        if not article_content.strip():

            st.warning(
                "Please enter an article before analyzing."
            )

        else:

            # -------------------------------------------------
            # TOKENIZATION
            # -------------------------------------------------

            sequence = tokenizer.texts_to_sequences(
                [article_content]
            )

            # -------------------------------------------------
            # PADDING
            # -------------------------------------------------

            padded = pad_sequences(
                sequence,
                maxlen=750,
                padding="post",
                truncating="post"
            )

            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            probability = model.predict(
                padded,
                verbose=0
            )[0][0]

            prediction = int(
                probability >= 0.5
            )


            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

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


            # -------------------------------------------------
            # DETAILS
            # -------------------------------------------------

            st.markdown("---")

            st.write("### Prediction Details")

            st.write(
                f"**Model probability (REAL):** "
                f"{probability:.4f}"
            )

            st.write(
                "**Classification threshold:** 0.50"
            )


            # -------------------------------------------------
            # DATASET LABEL
            # -------------------------------------------------

            if selected_sample != "Select a sample...":

                actual_label = sample_mapping[
                    selected_sample
                ]["label"]

                actual_label_text = (
                    "REAL"
                    if actual_label == 1
                    else "FAKE"
                )

                st.markdown("---")

                st.write(
                    "### Dataset Ground Truth"
                )

                st.info(
                    f"According to the original dataset, "
                    f"this article is labelled **{actual_label_text}**."
                )

    else:

        st.info(
            "Select an article from the dataset or enter "
            "your own news article and click **Analyze Authenticity**."
        )