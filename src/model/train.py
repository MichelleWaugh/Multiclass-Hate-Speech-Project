
"""
BERT Baseline Training Script
Project: Multiclass Hate Speech Detection and Severity Analysis using Transformers

ML Engineer:
    - Fine-tune bert-base-uncased
    - 5-class hate speech classification
    - Evaluate using accuracy, macro F1, and per-class F1
    - Save trained model and tokenizer

Expected input:
    data/processed/train.parquet
    data/processed/val.parquet
    data/processed/test.parquet

Expected columns:
    text
    label

Labels:
    0 -> not_hate
    1 -> weak_hate
    2 -> moderate_hate
    3 -> strong_hate
    4 -> extreme_hate

Output:
    models/bert_hate_v1/
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
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

WARMUP_STEPS = 500

WEIGHT_DECAY = 0.01

SEED = 42

TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
TEST_PATH = "data/processed/test.parquet"

OUTPUT_DIR = "models/bert_hate_v1"

LABEL_MAP_PATH = "data/label_map.json"


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

if torch.cuda.is_available():
    print("CUDA available: YES")
    print("GPU:", torch.cuda.get_device_name(0))
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
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

print("All Parquet files found.")
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

required_columns = {"text", "label"}

for split_name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{split_name} dataset is missing columns: "
            f"{missing_columns}"
        )

    if df["text"].isnull().any():
        raise ValueError(
            f"{split_name} dataset contains null text values."
        )

    if df["label"].isnull().any():
        raise ValueError(
            f"{split_name} dataset contains null labels."
        )

print("Data validation passed.")
print()


# ============================================================
# 7. LOAD LABEL MAP
# ============================================================

if os.path.exists(LABEL_MAP_PATH):

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        raw_label_map = json.load(f)

    print("Loaded label map:")
    print(raw_label_map)

else:

    print(
        "WARNING: label_map.json not found. "
        "Using default label mapping."
    )

    raw_label_map = {
        "0": "not_hate",
        "1": "weak_hate",
        "2": "moderate_hate",
        "3": "strong_hate",
        "4": "extreme_hate",
    }

print()


# ============================================================
# 8. CREATE LABEL MAPPINGS
# ============================================================

label2id = {
    label_name: int(label_id)
    for label_name, label_id in raw_label_map.items()
}

id2label = {
    label_id: label_name
    for label_name, label_id in label2id.items()
}

print("id2label:")
print(id2label)

print("\nlabel2id:")
print(label2id)

print()


# ============================================================
# 9. CONVERT PANDAS → HUGGINGFACE DATASET
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)

print("=" * 70)
print("HUGGINGFACE DATASETS")
print("=" * 70)

print(train_dataset)
print(val_dataset)
print(test_dataset)

print()


# ============================================================
# 10. LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = BertTokenizerFast.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")
print()


# ============================================================
# 11. TOKENIZATION
# ============================================================

def tokenize_function(examples):
    """
    Tokenize input text using BERT tokenizer.

    truncation=True:
        Cuts text if it exceeds MAX_LENGTH.

    max_length=MAX_LENGTH:
        Maximum number of tokens.

    padding=False:
        Dynamic padding will be handled later by
        DataCollatorWithPadding.
    """

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


print("=" * 70)
print("TOKENIZING DATA")
print("=" * 70)

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
# 12. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 13. LOAD BERT MODEL
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

print("Model loaded:")
print(MODEL_NAME)

print()


# ============================================================
# 14. COMPUTE METRICS
# ============================================================

def compute_metrics(eval_prediction):
    """
    Calculate evaluation metrics.

    Returns:
        accuracy
        macro F1
        per-class F1
    """

    predictions, labels = eval_prediction

    # predictions shape:
    # [number_of_samples, number_of_classes]

    predicted_labels = np.argmax(
        predictions,
        axis=-1
    )

    # Accuracy
    accuracy = accuracy_score(
        labels,
        predicted_labels
    )

    # Macro F1
    macro_f1 = f1_score(
        labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    # Per-class F1
    per_class_f1 = f1_score(
        labels,
        predicted_labels,
        average=None,
        labels=list(range(NUM_LABELS)),
        zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }

    # Add class-specific F1 scores
    for class_id, class_f1 in enumerate(per_class_f1):

        class_name = id2label.get(
            class_id,
            f"class_{class_id}"
        )

        metrics[f"f1_{class_name}"] = float(
            class_f1
        )

    return metrics


# ============================================================
# 15. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    # Where checkpoints/results are stored
    output_dir=OUTPUT_DIR,

    # Training hyperparameters
    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    warmup_steps=WARMUP_STEPS,

    # Evaluation
    eval_strategy="epoch",

    # Save after each epoch
    save_strategy="epoch",

    # Keep best model
    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    # Logging
    logging_strategy="steps",

    logging_steps=100,

    # Reproducibility
    seed=SEED,

    # Prevent excessive checkpoint storage
    save_total_limit=2,

    # Performance
    fp16=torch.cuda.is_available(),

    # Report nowhere by default
    report_to="none",

    # Keep output manageable
    disable_tqdm=False,
)


# ============================================================
# 16. CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    tokenizer=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ============================================================
# 17. TRAIN
# ============================================================

print("=" * 70)
print("STARTING BERT TRAINING")
print("=" * 70)

print()
print("Model:", MODEL_NAME)
print("Classes:", NUM_LABELS)
print("Learning rate:", LEARNING_RATE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", NUM_EPOCHS)
print("Max length:", MAX_LENGTH)
print("Warmup steps:", WARMUP_STEPS)
print("Weight decay:", WEIGHT_DECAY)
print()

train_result = trainer.train()


# ============================================================
# 18. SAVE MODEL
# ============================================================

print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to: {OUTPUT_DIR}")
print()


# ============================================================
# 19. VALIDATION EVALUATION
# ============================================================

print("=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

validation_metrics = trainer.evaluate(
    eval_dataset=val_dataset
)

for key, value in validation_metrics.items():
    print(f"{key}: {value}")

print()


# ============================================================
# 20. TEST EVALUATION
# ============================================================

print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_metrics = trainer.evaluate(
    eval_dataset=test_dataset,
    metric_key_prefix="test"
)

for key, value in test_metrics.items():
    print(f"{key}: {value}")

print()


# ============================================================
# 21. CLASSIFICATION REPORT ON TEST SET
# ============================================================

print("=" * 70)
print("TEST CLASSIFICATION REPORT")
print("=" * 70)

prediction_output = trainer.predict(
    test_dataset
)

test_predictions = np.argmax(
    prediction_output.predictions,
    axis=-1
)

test_labels = prediction_output.label_ids

target_names = [
    id2label[i]
    for i in range(NUM_LABELS)
]

report = classification_report(
    test_labels,
    test_predictions,
    labels=list(range(NUM_LABELS)),
    target_names=target_names,
    zero_division=0,
)

print(report)


# ============================================================
# 22. SAVE BASIC TRAINING RESULTS
# ============================================================

results = {
    "model": MODEL_NAME,
    "num_labels": NUM_LABELS,
    "max_length": MAX_LENGTH,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "warmup_steps": WARMUP_STEPS,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "validation_metrics": {
        key: float(value)
        for key, value in validation_metrics.items()
        if isinstance(value, (int, float))
    },
    "test_metrics": {
        key: float(value)
        for key, value in test_metrics.items()
        if isinstance(value, (int, float))
    },
    "classification_report": report,
}

results_path = os.path.join(
    OUTPUT_DIR,
    "training_results.json"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Model directory: {OUTPUT_DIR}")
print(f"Results file: {results_path}")

