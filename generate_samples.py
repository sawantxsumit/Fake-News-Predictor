import pandas as pd
from pathlib import Path
import pprint

PROJECT_DIR = Path(__file__).resolve().parent

FAKE_PATH = PROJECT_DIR / "data" / "Fake.csv"
REAL_PATH = PROJECT_DIR / "data" / "True.csv"

OUTPUT_PATH = PROJECT_DIR / "src" / "samples.py"


# LOAD DATA

fake_df = pd.read_csv(FAKE_PATH)
real_df = pd.read_csv(REAL_PATH)


# SELECT ACTUAL DATASET ARTICLES

# Select 4 real and 4 fake articles
real_samples = real_df.iloc[[0, 1, 2, 3]]
fake_samples = fake_df.iloc[[0, 1, 2, 3]]

samples = {}


# REAL ARTICLES

for i, (_, row) in enumerate(real_samples.iterrows(), 1):

    samples[f"Real {i}: {row['title']}"] = {
        "headline": str(row["title"]),
        "content": str(row["text"]),
        "actual_label": 1
    }


# FAKE ARTICLES

for i, (_, row) in enumerate(fake_samples.iterrows(), 1):

    samples[f"Fake {i}: {row['title']}"] = {
        "headline": str(row["title"]),
        "content": str(row["text"]),
        "actual_label": 0
    }


# CREATE A PYTHON FILE

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    f.write("# Automatically generated from ISOT dataset\n\n")
    f.write("SAMPLES = ")
    f.write(
        pprint.pformat(
            samples,
            width=100,
            sort_dicts=False
        )
    )

print(f"Created: {OUTPUT_PATH}")

print("\nSamples:")
for key, value in samples.items():
    print(
        f"{key} -> "
        f"{len(value['content']):,} characters"
    )