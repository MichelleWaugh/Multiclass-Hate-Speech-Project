from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# Setup
# ============================================================
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")

print("Loading dataset from Hugging Face...")

# Load dataset
ds = load_dataset("Shubhi324/measuring-hate-speech")
df = ds["train"].to_pandas()

print("Dataset loaded successfully!")

# ============================================================
# Automatically detect target columns
# ============================================================
TARGET_COLS = [c for c in df.columns if c.startswith("target_")]

# ============================================================
# 1. Basic dataset information
# ============================================================
print("\n" + "="*70)
print("BASIC DATASET INFORMATION")
print("="*70)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Target columns detected: {len(TARGET_COLS)}")

print("\nData types summary:")
print(df.dtypes.value_counts())

# Save schema
pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str)
}).to_csv(REPORTS_DIR / "column_schema.csv", index=False)

# ============================================================
# 2. Comment-level analysis
# ============================================================
print("\n" + "="*70)
print("COMMENT-LEVEL ANALYSIS")
print("="*70)

unique_comments = df["text"].nunique()
duplicate_rows = len(df) - unique_comments

print(f"Total rows: {len(df):,}")
print(f"Unique comments: {unique_comments:,}")
print(f"Repeated comment rows: {duplicate_rows:,}")

annotation_counts = df.groupby("comment_id").size()

print("\nAnnotations per comment statistics:")
print(annotation_counts.describe())

annotation_counts.value_counts().sort_index().to_csv(
    REPORTS_DIR / "annotation_count_distribution.csv"
)

# ============================================================
# 3. Hate speech score distribution
# ============================================================
print("\n" + "="*70)
print("HATE SPEECH SCORE DISTRIBUTION")
print("="*70)

print(df["hate_speech_score"].describe())

plt.figure(figsize=(8,5))
sns.histplot(df["hate_speech_score"], bins=40, kde=True)
plt.title("Distribution of Hate Speech Score")
plt.xlabel("Hate Speech Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "hate_score_distribution.png", dpi=300)
plt.close()

# ============================================================
# 4. Create severity labels (for analysis only)
# ============================================================
print("\n" + "="*70)
print("SEVERITY LABEL DISTRIBUTION")
print("="*70)

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

df["severity_label"] = df["hate_speech_score"].apply(score_to_label)

label_distribution = df["severity_label"].value_counts()

print(label_distribution)

label_distribution.to_csv(
    REPORTS_DIR / "severity_label_distribution.csv"
)

# ============================================================
# 5. Text length analysis
# ============================================================
print("\n" + "="*70)
print("TEXT LENGTH ANALYSIS")
print("="*70)

df["word_count"] = df["text"].astype(str).apply(
    lambda x: len(x.split())
)
df["char_count"] = df["text"].astype(str).apply(len)

print(df[["word_count", "char_count"]].describe())

length_stats = (
    df.groupby("severity_label")[["word_count", "char_count"]]
    .mean()
)

length_stats.to_csv(REPORTS_DIR / "text_length_by_label.csv")

# ============================================================
# 6. Target group analysis (all target_* columns)
# ============================================================
print("\n" + "="*70)
print("TARGET GROUP ANALYSIS")
print("="*70)

target_summary = []

for col in TARGET_COLS:
    target_summary.append({
        "target_group": col,
        "mean_score": float(df[col].mean()),
        "non_zero_comments": int((df[col] > 0).sum())
    })

target_df = (
    pd.DataFrame(target_summary)
    .sort_values("mean_score", ascending=False)
)

print(target_df.head(20))

target_df.to_csv(
    REPORTS_DIR / "target_group_summary.csv",
    index=False
)

plt.figure(figsize=(10,8))
sns.barplot(
    data=target_df.head(20),
    x="mean_score",
    y="target_group"
)
plt.title("Top 20 Target Groups by Average Score")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "top_target_groups.png", dpi=300)
plt.close()

# ============================================================
# 7. Missing values
# ============================================================
print("\n" + "="*70)
print("MISSING VALUES")
print("="*70)

missing = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing.head(20))
    missing.to_csv(REPORTS_DIR / "missing_values.csv")

# ============================================================
# 8. Correlation with hate speech score
# ============================================================
print("\n" + "="*70)
print("TOP TARGET CORRELATIONS WITH HATE SCORE")
print("="*70)

corr = (
    df[TARGET_COLS + ["hate_speech_score"]]
    .corr()["hate_speech_score"]
    .drop("hate_speech_score")
    .sort_values(ascending=False)
)

print(corr.head(20))

corr.to_csv(REPORTS_DIR / "target_correlations.csv")

# ============================================================
# 9. Save EDA summary
# ============================================================
summary = {
    "rows": len(df),
    "columns": len(df.columns),
    "unique_comments": unique_comments,
    "duplicate_rows": duplicate_rows,
    "avg_annotations_per_comment": float(annotation_counts.mean()),
    "max_annotations_per_comment": int(annotation_counts.max()),
    "avg_word_count": float(df["word_count"].mean()),
    "avg_char_count": float(df["char_count"].mean()),
    "hate_score_mean": float(df["hate_speech_score"].mean()),
    "hate_score_std": float(df["hate_speech_score"].std()),
    "num_target_columns": len(TARGET_COLS),
}

pd.DataFrame([summary]).to_csv(
    REPORTS_DIR / "eda_summary.csv",
    index=False
)

print("\n" + "="*70)
print("EDA COMPLETED SUCCESSFULLY")
print("="*70)
print("Key finding: The dataset is annotation-level (multiple annotations per comment).")
print("Next step: Aggregate to one row per comment for preprocessing.")