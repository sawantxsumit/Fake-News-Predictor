"""
Run this from the same folder as predictor.py (e.g. `python diagnose_predictions.py`).
It bypasses Streamlit entirely and checks the model + tokenizer directly, so we can see
whether the problem is a file mismatch, a vocabulary problem, or the model itself.
"""

from predictor import TokenizerPredictor, clean_text

REAL_SAMPLES = [
    ("Breakthrough in Battery Tech",
     "Researchers at MIT have developed a new solid-state battery architecture that could "
     "double the range of electric vehicles. The new design uses common materials and is "
     "reportedly safer than current lithium-ion cells, with mass production expected by 2027."),
    ("Eurozone Inflation Drops",
     "Inflation in the Eurozone fell to 2.4% last month, its lowest level in two years, "
     "prompting speculation that the European Central Bank may cut interest rates sooner "
     "than expected. Energy prices were the primary driver of the decline."),
]

FAKE_SAMPLES = [
    ("Senator Caught in Alien Pact",
     "Leaked audio proves that a senior Senator signed a secret treaty with the Galactic "
     "Federation to trade human water for advanced mining technology. Mainstream media is "
     "blacking out this story to protect the elite globalist agenda!"),
    ("Government Giving Away Bitcoin",
     "A new federal stimulus program allows every citizen to claim 0.5 Bitcoin instantly. "
     "This is a limited-time offer to boost the digital economy. Click the link below and "
     "enter your wallet keys to receive your government-mandated crypto grant!"),
]

print("=" * 70)
print("LOADING PREDICTOR")
print("=" * 70)
predictor = TokenizerPredictor()

print()
print("=" * 70)
print("MODEL / TOKENIZER SANITY CHECK")
print("=" * 70)
expected_len = predictor.model.input_shape[1]
print(f"Sequence length the MODEL was built/trained with : {expected_len}")
print(f"Sequence length predictor.py is currently using   : {predictor.max_len}")
if expected_len is not None and expected_len != predictor.max_len:
    print(">>> MISMATCH! predictor.py's max_len does not match the model. "
          "Re-download model_config.json from the training run and confirm "
          "it's paired with the SAME model.keras / tokenizer.pickle.")
else:
    print("OK: lengths match (or model accepts variable length).")

print(f"\nTokenizer vocabulary size (word_index) : {len(predictor.tokenizer.word_index)}")
print(f"Tokenizer num_words setting             : {predictor.tokenizer.num_words}")
oov_token = getattr(predictor.tokenizer, "oov_token", None)
oov_index = predictor.tokenizer.word_index.get(oov_token) if oov_token else None
print(f"OOV token / index                       : {oov_token} / {oov_index}")

print()
print("=" * 70)
print("PER-SAMPLE CHECK")
print("=" * 70)

def check(label_expected, title, text):
    combined = predictor._compose(title, text)
    cleaned = clean_text(combined)
    tokens = cleaned.split()
    seq = predictor.tokenizer.texts_to_sequences([cleaned])[0]
    oov_count = sum(1 for t in seq if oov_index is not None and t == oov_index)
    zero_count = predictor.max_len - len(seq[:predictor.max_len])  # padding added

    result = predictor.predict_single_news(title, text)

    print(f"\n[{label_expected}] {title}")
    print(f"  cleaned tokens      : {len(tokens)}  (first 12: {tokens[:12]})")
    print(f"  sequence length     : {len(seq)}  | OOV tokens: {oov_count}  | pad zeros added: {max(zero_count,0)}")
    print(f"  -> predicted        : {result['label_str'].upper()}  (probability={result['probability']:.4f})")

for title, text in REAL_SAMPLES:
    check("EXPECTED REAL", title, text)

for title, text in FAKE_SAMPLES:
    check("EXPECTED FAKE", title, text)

print()
print("=" * 70)
print("HOW TO READ THIS")
print("=" * 70)
print("""
1. If the two sequence-length lines under MODEL/TOKENIZER didn't match -> that's the bug,
   fix that first (re-pair model/tokenizer/config from the SAME training run).

2. If most tokens for every sample show as OOV, or 'cleaned tokens' is near 0 for
   normal-looking sentences -> the tokenizer doesn't recognize this vocabulary, meaning
   it's probably not the tokenizer that was actually fit on your training text (wrong file,
   or fit on different preprocessing). Every input then looks almost identical to the
   model (mostly the OOV token), which explains a constant prediction.

3. If tokens/OOV counts look fine and lengths match, but probability is high (>0.8) for
   BOTH the real AND fake samples above -> the tokenizer/serving pipeline is fine, and the
   model itself likely didn't learn to separate the classes. In that case, go back to the
   training notebook and check the classification_report/confusion matrix it printed on
   its own held-out test set:
     - If THAT was also stuck near 50-55% or biased to one class, the model needs to be
       retrained (check: did class_weight get applied? did val_loss actually decrease over
       epochs, or did it plateau immediately?).
     - If the notebook itself reported good test accuracy (e.g. 90%+), then something is
       different between the model file you're running here and the one that was
       evaluated in the notebook — re-download fresh model.keras/tokenizer.pickle/
       model_config.json as a matched set and replace all three at once.
""")
