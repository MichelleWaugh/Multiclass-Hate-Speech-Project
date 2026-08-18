"""
BERT Baseline Training Script
Project: Multiclass Hate Speech Detection and Severity Analysis using Transformers

ML Engineer:
    - Fine-tune bert-base-uncased
    - 5-class hate speech severity classification
    - Evaluate accuracy, precision, recall, macro F1, weighted F1
    - Save trained model, tokenizer, metrics, and classification report

Expected project structure:
    project_root/
    ├── data/
    │   ├── label_map.json
    │   └── processed/
    │       ├── train.parquet
    │       ├── val.parquet
    │       └── test.parquet
    ├── src/
    │   └── model/
    │       └── train.py
    └── models/
        └── bert_hate_v1/

Supported label_map.json formats:
    {"not_hate": 0, "weak_hate": 1, ...}
or:
    {"0": "not_hate", "1": "weak_hate", ...}

Expected final label IDs:
    0 -> not_hate
    1 -> weak_hate
    2 -> moderate_hate
    3 -> strong_hate
    4 -> extreme_hate
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"

NUM_LABELS = 5
MAX_LENGTH = 128

LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 4

# If the T4 gives CUDA OOM, change:
# BATCH_SIZE = 8
# GRADIENT_ACCUMULATION_STEPS = 2
GRADIENT_ACCUMULATION_STEPS = 1

WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42

# ------------------------------------------------------------
# Resolve paths from the repository root instead of relying
# on the current working directory.
#
# train.py is expected at:
# project_root/src/model/train.py
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"

LABEL_MAP_PATH = PROJECT_ROOT / "data" / "label_map.json"

OUTPUT_DIR = PROJECT_ROOT / "models" / "bert_hate_v1"


DEFAULT_LABEL_MAP = {
    "not_hate": 0,
    "weak_hate": 1,
    "moderate_hate": 2,
    "strong_hate": 3,
    "extreme_hate": 4,
}


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. DEVICE INFORMATION
# ============================================================

print("=" * 70)
print("DEVICE INFORMATION")
print("=" * 70)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    print("CUDA available: YES")
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )
else:
    print("CUDA available: NO")
    print("Training will use CPU.")

print()


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

print("=" * 70)
print("CHECKING DATA FILES")
print("=" * 70)

required_files = [
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print("All Parquet files found.")

if not LABEL_MAP_PATH.exists():
    print(
        f"WARNING: {LABEL_MAP_PATH} not found. "
        "Using the default label mapping."
    )

print()


# ============================================================
# 5. LOAD PARQUET DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)
test_df = pd.read_parquet(TEST_PATH)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nTrain label distribution:")
print(train_df["label"].value_counts().sort_index())

print()


# ============================================================
# 6. BASIC DATA VALIDATION
# ============================================================

print("=" * 70)
print("VALIDATING DATA")
print("=" * 70)

required_columns = {"text", "label"}

for split_name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{split_name} dataset is missing columns: {missing_columns}"
        )

    if df["text"].isnull().any():
        raise ValueError(
            f"{split_name} dataset contains null text values."
        )

    if df["label"].isnull().any():
        raise ValueError(
            f"{split_name} dataset contains null labels."
        )

    # Convert labels to integers if possible.
    try:
        df["label"] = df["label"].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{split_name} labels must be integer IDs 0-{NUM_LABELS - 1}. "
            f"Found dtype: {df['label'].dtype}"
        ) from exc

    invalid_labels = sorted(
        set(df["label"].unique()) - set(range(NUM_LABELS))
    )

    if invalid_labels:
        raise ValueError(
            f"{split_name} contains invalid labels: {invalid_labels}. "
            f"Expected labels: 0-{NUM_LABELS - 1}."
        )

print("Data validation passed.")
print()


# ============================================================
# 7. LOAD AND NORMALIZE LABEL MAP
# ============================================================

print("=" * 70)
print("LOADING LABEL MAP")
print("=" * 70)


def load_label_map(path: Path) -> dict:
    """
    Load either of these formats:

        {"not_hate": 0, "weak_hate": 1, ...}

    or:

        {"0": "not_hate", "1": "weak_hate", ...}

    Always return:
        label2id = {"not_hate": 0, ...}
    """
    if not path.exists():
        print("Using default label mapping:")
        print(DEFAULT_LABEL_MAP)
        return DEFAULT_LABEL_MAP.copy()

    with open(path, "r", encoding="utf-8") as file:
        raw_map = json.load(file)

    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError("label_map.json must contain a non-empty JSON object.")

    label2id = {}

    # Format A:
    # {"not_hate": 0, "weak_hate": 1, ...}
    if all(
        isinstance(key, str) and not key.lstrip("-").isdigit()
        for key in raw_map.keys()
    ):
        for label_name, label_id in raw_map.items():
            try:
                label2id[str(label_name)] = int(label_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid label ID for '{label_name}': {label_id}"
                ) from exc

    # Format B:
    # {"0": "not_hate", "1": "weak_hate", ...}
    else:
        for label_id, label_name in raw_map.items():
            try:
                numeric_id = int(label_id)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric label ID: {label_id}"
                ) from exc

            label2id[str(label_name)] = numeric_id

    return label2id


label2id = load_label_map(LABEL_MAP_PATH)

expected_label2id = DEFAULT_LABEL_MAP

if label2id != expected_label2id:
    raise ValueError(
        "The label mapping does not match the project's expected "
        f"5-class mapping.\n"
        f"Expected: {expected_label2id}\n"
        f"Found:    {label2id}"
    )

id2label = {
    label_id: label_name
    for label_name, label_id in label2id.items()
}

print("label2id:")
print(label2id)

print("\nid2label:")
print(id2label)

print()


# ============================================================
# 8. CONVERT PANDAS -> HUGGING FACE DATASETS
# ============================================================

print("=" * 70)
print("CREATING HUGGING FACE DATASETS")
print("=" * 70)

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False,
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]],
    preserve_index=False,
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False,
)

print(train_dataset)
print(val_dataset)
print(test_dataset)

print()


# ============================================================
# 9. LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

print("Tokenizer loaded:", MODEL_NAME)
print()


# ============================================================
# 10. TOKENIZATION
# ============================================================

print("=" * 70)
print("TOKENIZING DATA")
print("=" * 70)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing training data",
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing validation data",
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test data",
)

print("Tokenization completed.")
print()


# ============================================================
# 11. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
)


# ============================================================
# 12. LOAD BERT MODEL
# ============================================================

print("=" * 70)
print("LOADING BERT MODEL")
print("=" * 70)

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

print("Model loaded:", MODEL_NAME)
print("Number of labels:", NUM_LABELS)
print()


# ============================================================
# 13. METRICS
# ============================================================

def compute_metrics(eval_prediction):
    """
    Metrics used during validation and model selection.
    """

    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_labels = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, predicted_labels)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )

    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
    }


# ============================================================
# 14. TRAINING ARGUMENTS
# ============================================================

print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print("Model:", MODEL_NAME)
print("Classes:", NUM_LABELS)
print("Max length:", MAX_LENGTH)
print("Batch size:", BATCH_SIZE)
print("Gradient accumulation:", GRADIENT_ACCUMULATION_STEPS)
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Warmup ratio:", WARMUP_RATIO)
print("Output:", OUTPUT_DIR)

print()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    # Hyperparameters
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,

    # Evaluation and checkpointing
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    # Logging
    logging_strategy="steps",
    logging_steps=100,
    report_to="none",

    # Reproducibility
    seed=SEED,

    # Checkpoint management
    save_total_limit=2,

    # T4-friendly mixed precision
    fp16=torch.cuda.is_available(),

    # Keep progress visible
    disable_tqdm=False,
)


# ============================================================
# 15. CREATE TRAINER
# ============================================================

print("=" * 70)
print("CREATING TRAINER")
print("=" * 70)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    # Newer Transformers versions use processing_class.
    processing_class=tokenizer,

    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Trainer created successfully.")
print()


# ============================================================
# 16. TRAIN
# ============================================================

print("=" * 70)
print("STARTING BERT TRAINING")
print("=" * 70)

print()

try:
    train_result = trainer.train()
except torch.cuda.OutOfMemoryError as exc:
    print()
    print("=" * 70)
    print("CUDA OUT OF MEMORY")
    print("=" * 70)
    print(
        "The T4 did not have enough GPU memory for the current batch size."
    )
    print()
    print("Recommended fallback:")
    print("    BATCH_SIZE = 8")
    print("    GRADIENT_ACCUMULATION_STEPS = 2")
    print()
    print("This gives an effective batch size of approximately 16.")
    print("=" * 70)
    raise exc


# ============================================================
# 17. SAVE TRAINING STATE + MODEL + TOKENIZER
# ============================================================

print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

# Save trainer state as well.
trainer.save_state()

# Save training metrics.
train_metrics = train_result.metrics
trainer.log_metrics("train", train_metrics)
trainer.save_metrics("train", train_metrics)

print(f"Model saved to: {OUTPUT_DIR}")
print()


# ============================================================
# 18. VALIDATION EVALUATION
# ============================================================

print("=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

validation_metrics = trainer.evaluate(
    eval_dataset=val_dataset,
)

trainer.log_metrics("validation", validation_metrics)
trainer.save_metrics("validation", validation_metrics)

for key, value in validation_metrics.items():
    print(f"{key}: {value}")

print()


# ============================================================
# 19. TEST EVALUATION
# ============================================================

print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_metrics = trainer.evaluate(
    eval_dataset=test_dataset,
    metric_key_prefix="test",
)

trainer.log_metrics("test", test_metrics)
trainer.save_metrics("test", test_metrics)

for key, value in test_metrics.items():
    print(f"{key}: {value}")

print()


# ============================================================
# 20. TEST CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("TEST CLASSIFICATION REPORT")
print("=" * 70)

prediction_output = trainer.predict(test_dataset)

test_predictions = prediction_output.predictions

if isinstance(test_predictions, tuple):
    test_predictions = test_predictions[0]

test_predictions = np.argmax(
    test_predictions,
    axis=-1,
)

test_labels = prediction_output.label_ids

target_names = [
    id2label[i]
    for i in range(NUM_LABELS)
]

report_text = classification_report(
    test_labels,
    test_predictions,
    labels=list(range(NUM_LABELS)),
    target_names=target_names,
    zero_division=0,
)

report_dict = classification_report(
    test_labels,
    test_predictions,
    labels=list(range(NUM_LABELS)),
    target_names=target_names,
    zero_division=0,
    output_dict=True,
)

print(report_text)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=list(range(NUM_LABELS)),
)

print(cm)

confusion_matrix_path = OUTPUT_DIR / "confusion_matrix.csv"

cm_df = pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names,
)

cm_df.to_csv(confusion_matrix_path)

print(f"\nConfusion matrix saved to: {confusion_matrix_path}")
print()


# ============================================================
# 22. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = OUTPUT_DIR / "classification_report.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report_dict, file, indent=4)

report_txt_path = OUTPUT_DIR / "classification_report.txt"

with open(report_txt_path, "w", encoding="utf-8") as file:
    file.write(report_text)

print(f"Classification report saved to: {report_path}")
print(f"Classification report text saved to: {report_txt_path}")
print()


# ============================================================
# 23. SAVE COMPLETE TRAINING RESULTS
# ============================================================

def make_json_serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


results = {
    "model": MODEL_NAME,
    "num_labels": NUM_LABELS,
    "id2label": {
        str(key): value
        for key, value in id2label.items()
    },
    "label2id": label2id,
    "max_length": MAX_LENGTH,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": (
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    ),
    "epochs": NUM_EPOCHS,
    "warmup_ratio": WARMUP_RATIO,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "device": DEVICE,
    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ),
    "train_metrics": {
        key: make_json_serializable(value)
        for key, value in train_metrics.items()
    },
    "validation_metrics": {
        key: make_json_serializable(value)
        for key, value in validation_metrics.items()
    },
    "test_metrics": {
        key: make_json_serializable(value)
        for key, value in test_metrics.items()
    },
    "classification_report": report_dict,
}

results_path = OUTPUT_DIR / "training_results.json"

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Model directory: {OUTPUT_DIR}")
print(f"Results file: {results_path}")
print(f"Classification report: {report_path}")
print(f"Confusion matrix: {confusion_matrix_path}")
print("=" * 70)