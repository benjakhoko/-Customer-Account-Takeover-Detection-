"""
sample_dataset.py
Reads the full RBA dataset in chunks and creates a smaller,
class-balanced sample that keeps enough Account Takeover cases
for training.
"""
import pandas as pd

FULL_CSV = "dataset/rba-dataset.csv"          # full file from Zenodo
OUT_CSV  = "dataset/dataset_sample.csv"   # output sample

CHUNK_SIZE = 500_000
TARGET = "Is Account Takeover"

ato_rows   = []
normal_rows = []
MAX_NORMAL = 50_000      # cap normal rows
MAX_ATO    = 50_000      # cap ATO rows (keep all if fewer)

for chunk in pd.read_csv(FULL_CSV, chunksize=CHUNK_SIZE, low_memory=False):
    chunk[TARGET] = chunk[TARGET].astype(bool)
    ato_rows.append(chunk[chunk[TARGET]])
    if len(normal_rows) < MAX_NORMAL:
        normal_rows.append(chunk[~chunk[TARGET]])

ato_df    = pd.concat(ato_rows, ignore_index=True).head(MAX_ATO)
normal_df = pd.concat(normal_rows, ignore_index=True).head(MAX_NORMAL)

sample = pd.concat([ato_df, normal_df], ignore_index=True).sample(
    frac=1, random_state=42
).reset_index(drop=True)

sample.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print("Shape:", sample.shape)
print("Target counts:")
print(sample[TARGET].value_counts())