import streamlit as st
import pandas as pd
import time
import os
from predictor import TokenizerPredictor

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Fake News Predictor",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
   /* Increase header font size */
      h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
    }
    
    /* Increase font size of tab labels */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.4rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# LOAD PREDICTOR 
@st.cache_resource
def get_predictor():
    try:
        return TokenizerPredictor()
    except Exception as e:
        st.error(f"Failed to initialize model: {e}")
        return None

predictor = get_predictor()



#  SAMPLE NEWS
SAMPLES = {
    "Select a sample...": {"title": "", "text": ""},
    
    # REAL NEWS SAMPLES
    "Real: Tech Innovation": {
        "title": "Breakthrough in Battery Tech",
        "text": "Researchers at MIT have developed a new solid-state battery architecture that could double the range of electric vehicles. The new design uses common materials and is reportedly safer than current lithium-ion cells, with mass production expected by 2027."
    },
    "Real: Global Economy": {
        "title": "Eurozone Inflation Drops",
        "text": "Inflation in the Eurozone fell to 2.4% last month, its lowest level in two years, prompting speculation that the European Central Bank may cut interest rates sooner than expected. Energy prices were the primary driver of the decline."
    },
    "Real: Health Science": {
        "title": "New Malaria Vaccine Approved",
        "text": "The World Health Organization has prequalified a second malaria vaccine, R21/Matrix-M, marking a significant milestone in global health. Early trials show the vaccine is highly effective and can be manufactured at a lower cost than previous options."
    },
    "Real: Space Exploration": {
        "title": "SpaceX Starship Launch",
        "text": "SpaceX successfully launched its massive Starship rocket for a third test flight today. While the booster separation was successful, the team lost contact with the vehicle during re-entry. Data collected will be vital for future lunar missions."
    },

    # --- FAKE NEWS SAMPLES ---
    "Fake: Political Scandal": {
        "title": "Senator Caught in Alien Pact",
        "text": "Leaked audio proves that a senior Senator signed a secret treaty with the Galactic Federation to trade human water for advanced mining technology. Mainstream media is blacking out this story to protect the elite globalist agenda!"
    },
    "Fake: Miracle Diet": {
        "title": "Eat This to Melt Fat Overnight",
        "text": "Doctors are furious about this 5-second 'rice hack' that burns 20lbs of belly fat in a week. No exercise needed! Big Pharma has been hiding this ancient enzyme for decades because it would bankrupt the weight loss industry."
    },
    "Fake: Celebrity Hoax": {
        "title": "Famous Actor Faked Death",
        "text": "BREAKING: A Hollywood insider reveals that a beloved action star who 'died' in 2013 is actually alive and living in a bunker in New Zealand. Photos show him directing a secret documentary exposing the Illuminati."
    },
    "Fake: Crypto Scam": {
        "title": "Government Giving Away Bitcoin",
        "text": "A new federal stimulus program allows every citizen to claim 0.5 Bitcoin instantly. This is a limited-time offer to boost the digital economy. Click the link below and enter your wallet keys to receive your government-mandated crypto grant!"
    }
}

# SESSION STATE MANAGEMENT
if "news_title" not in st.session_state: st.session_state.news_title = ""
if "news_text" not in st.session_state: st.session_state.news_text = ""

def update_fields():
    selected = st.session_state.sample_selector
    if selected != "Select a sample...":
        st.session_state.news_title = SAMPLES[selected]["title"]
        st.session_state.news_text = SAMPLES[selected]["text"]

#  UI LAYOUT 

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964063.png", width=70)
    st.title("News Authenticator")
    st.caption("v1.0.0 | LSTM Model")
    
    st.markdown("---")
    
    # Section 1: How it Works
    with st.expander("🛠️ How it Works", expanded=True):
        st.markdown("""
        **1. Preprocessing** Removes URLs, special characters, and converts text to lowercase. 
        
        **2. Tokenization** Converts words into numerical sequences using a pre-trained tokenizer (mapped to Word2Vec).
        
        **3. Padding** Ensures all inputs are the same length (1000 tokens) for the neural network.
        
        **4. Deep Learning** A simple **LSTM** model processes the sequence to understand context.
        
        **5. Classification** The model outputs a probability score (0-1).
        """)
        
    # Section 2: Tech Stack
    with st.expander("💻 Tech Stack", expanded=False):
        st.markdown("""
        * **Python 3.10+**
        * **TensorFlow/Keras:** Model training
        * **Streamlit:** UI Framework
        * **NLTK:** Text processing
        * **Pandas:** Data handling
        """)

    # Section 3: Dataset Info
    with st.expander("📊 Dataset Details", expanded=False):
        st.info("""
        **WELFake News Dataset**
        * ~70,000 Articles
        * Balanced (50% Real / 50% Fake)
        * Sources: Reuters (Real) vs Flagged Sites (Fake)
        """)


# Main Header
st.title("🕵️‍♂️ Fake News Predictor")
st.markdown("Verify the authenticity of news articles using Deep learning model ")

# Tabs for Single vs Batch
tab1, tab2 = st.tabs(["🔍 Analyze Single Article", "📂 Batch Analysis"])

#  SINGLE PREDICTION
with tab1:
    col_input, col_result = st.columns([1.5, 1], gap="large")

    with col_input:
        st.subheader("Input Details")
        
        # Sample Selector
        st.selectbox(
            "Quick Load sample news (Optional):", 
            options=list(SAMPLES.keys()), 
            key="sample_selector",
            on_change=update_fields
        )

        # Inputs
        title_in = st.text_input("Headline (Optional):", key="news_title", placeholder="e.g. Breaking News...")
        text_in = st.text_area("Article Content:", key="news_text", height=250, placeholder="Paste the full body text here...")

        if st.button("Analyze Authenticity", type="primary", use_container_width=True):
            if not text_in.strip():
                st.warning("⚠️ Article content is required.")
            elif predictor is None:
                st.error("❌ Model not loaded.")
            else:
               # Animation
                with col_result:
                    status_box = st.empty()
                    with status_box.container():
                        st.markdown("#### 🔄 Processing...")
                        prog_bar = st.progress(0)
                        
                        steps = [
                            "Loading LSTM vectors...",
                            "Scanning linguistic patterns...",
                        ]
                        
                        for i, step in enumerate(steps):
                            # Update text and progress bar
                            st.caption(f"Step {i+1}/{len(steps)}: {step}")
                            # Smooth fill
                            start = i * 25
                            end = (i + 1) * 25
                            for p in range(start, end + 1, 5):
                                prog_bar.progress(p)
                                time.sleep(0.06) 
                        
                        time.sleep(0.2) # Short pause at 100%
                    
                    # Clear animation container
                    status_box.empty()
                
                    result = predictor.predict_single_news(title_in, text_in)
                    
                    # Store result in session state
                    st.session_state.last_result = result

    # Result Column
    with col_result:
        st.subheader("Analysis Results")
        
        if "last_result" in st.session_state:
            res = st.session_state.last_result
            label_str = res['label_str']
            prob = res['probability']
            
            # Create a card-like container
            with st.container(border=True):
                is_real = label_str == "real"
                
                if is_real:
                    st.success("## ✅ REAL")
                    msg = "This content aligns with patterns found in reliable news."
                    bar_color = "green"
                else:
                    st.error("## 🚨 FAKE")
                    msg = "This content exhibits patterns common in misinformation."
                    bar_color = "red"
                
                st.write(msg)
                st.divider()
                
                # Metrics
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Classification", label_str.upper())
                with m2:
                    # Confidence is p if Real, else 1-p
                    conf = prob if is_real else (1.0 - prob)
                    st.metric("Confidence", f"{conf:.1%}")
                
                # st.caption("Authenticity Score (0=Fake, 1=Real)")
                st.progress(prob)

# BATCH PROCESSING
with tab2:
    st.subheader("Bulk Analysis")
    st.write("Upload a CSV file containing a column named `text` (and optionally `title`) to analyze multiple articles at once.")
    
    #  SAMPLE CSV DOWNLOADER 
    csv_path = "test_batch.csv"
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button(
                label="📄 Download Example CSV to Test",
                data=f,
                file_name="test_news_batch.csv",
                mime="text/csv",
                help="Click to download a pre-made CSV with 10 mixed news samples."
            )
    else:
        st.warning("Example CSV file not found in directory.")

    st.divider()
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and predictor:
        df = pd.read_csv(uploaded_file)
        
        # Validate columns
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            st.success(f"Loaded {len(df)} rows.")
            
            if st.button("Run Batch Prediction"):
                with st.status("Processing batch...", expanded=True) as status:
                    # Prepare text list (combine title + text if title exists)
                    texts_to_process = []
                    for _, row in df.iterrows():
                        t_title = row['title'] if 'title' in df.columns else ""
                        t_text = row['text']
                
                        combined = str(t_title) + " " + str(t_text)
                        texts_to_process.append(combined)
                    
                    st.write("Vectorizing text...")
                    results = predictor.predict_batch(texts_to_process)
                    
                    st.write("Applying labels...")
                    # Add results back to DataFrame
                    df['prediction'] = [r['label_str'] for r in results]
                    df['confidence_real'] = [r['probability'] for r in results]
                    
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                # Display Result
                st.markdown("Displaying first 10 rows :")
                st.dataframe(df.head(10))
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Analyzed CSV",
                    csv,
                    "analyzed_news.csv",
                    "text/csv",
                    key='download-csv'
                )

st.markdown("---")
with st.expander("How this model works"):
    st.write(
        """
        1. Text is lowercased and cleaned (URLs, HTML, non-letters removed).
        2. Stopwords removed and words lemmatized.
        3. Tokenizer -> padded sequences (maxlen=1000).
        4. LSTM model outputs probability that text is Fake (class=1).
        """
    )