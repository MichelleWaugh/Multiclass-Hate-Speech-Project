from datasets import load_dataset
import pandas as pd
import re
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# Setup
# ============================================================
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")

# Load dataset
ds = load_dataset("Shubhi324/measuring-hate-speech")
df = ds["train"].to_pandas()

print(f"Original shape: {df.shape}")

# ============================================================
# Detect target columns
# ============================================================
TARGET_COLS = [c for c in df.columns if c.startswith("target_")]

print(f"Target columns detected: {len(TARGET_COLS)}")

# ============================================================
# Aggregate to one row per comment
# ============================================================

aggregation = {
    "text": "first",
    "hate_speech_score": "mean"
}

for col in TARGET_COLS:
    aggregation[col] = "max"

comment_df = (
    df.groupby("comment_id")
      .agg(aggregation)
      .reset_index()
)

print(f"Aggregated shape: {comment_df.shape}")

# ============================================================
# Create severity labels
# ============================================================

def score_to_label(score):
    if score < -1:
        return "not_hate"
    elif score <= 0:
        return "weak_hate"
    elif score <= 1:
        return "moderate_hate"
    elif score <= 2:
        return "strong_hate"
    else:
        return "extreme_hate"

comment_df["severity_label"] = (
    comment_df["hate_speech_score"]
    .apply(score_to_label)
)

label_map = {
    "not_hate": 0,
    "weak_hate": 1,
    "moderate_hate": 2,
    "strong_hate": 3,
    "extreme_hate": 4
}

comment_df["label"] = (
    comment_df["severity_label"]
    .map(label_map)
)

print("\nSeverity label distribution:")
print(comment_df["severity_label"].value_counts())

# ============================================================
# Minimal text cleaning for mBERT
# ============================================================

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\\w+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\\s+", " ", text).strip()

    return text

# Apply cleaning directly to the text column
comment_df["text"] = (
    comment_df["text"]
    .astype(str)
    .apply(clean_text)
)

# ============================================================
# Save label map
# ============================================================

with open("data/label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

# ============================================================
# Compute class weights
# ============================================================

classes = np.array(sorted(comment_df["label"].unique()))

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=comment_df["label"]
)

class_weights = {
    str(int(cls)): float(weight)
    for cls, weight in zip(classes, weights)
}

with open("data/class_weights.json", "w") as f:
    json.dump(class_weights, f, indent=4)

print("\nClass weights:")
print(class_weights)

# ============================================================
# Stratified train / validation / test split
# ============================================================

train_df, temp_df = train_test_split(
    comment_df,
    test_size=0.2,
    stratify=comment_df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print("\nSplit sizes:")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

# ============================================================
# Save datasets
# ============================================================

comment_df.to_parquet(
    PROCESSED_DIR / "comment_level_dataset.parquet",
    index=False
)

train_df.to_parquet(
    PROCESSED_DIR / "train.parquet",
    index=False
)

val_df.to_parquet(
    PROCESSED_DIR / "val.parquet",
    index=False
)

test_df.to_parquet(
    PROCESSED_DIR / "test.parquet",
    index=False
)

print("\nFiles saved:")
print("- comment_level_dataset.parquet")
print("- train.parquet")
print("- val.parquet")
print("- test.parquet")
print("- data/class_weights.json")
print("- data/label_map.json")

print("\nPreprocessing completed successfully!")