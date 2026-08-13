# Multiclass hate speech detection and severity analysis using transformers

## Project overview

This project focuses on multiclass hate speech detection and severity analysis using transformer-based models (mBERT). The objective is to classify online comments into five severity levels while preserving a broad spectrum of hate-target categories across race, religion, gender, sexuality, origin, age, and disability, education, ideology, sarcasm, etc.

## Dataset

Source: `Shubhi324/measuring-hate-speech` (Hugging Face)

Original dataset:

* 135,556 annotation rows
* Multiple annotators per comment
* 53 target category columns

Processed dataset:

* 39,565 unique comments
* 53 preserved target dimensions
* 5 severity classes

## Severity classes

* Not hate
* Weak hate
* Moderate hate
* Strong hate
* Extreme hate

## Data pipeline

The preprocessing pipeline performs:

* aggregation of multiple annotator rows by `comment_id`
* averaging of `hate_speech_score`
* preservation of all target categories using max aggregation
* severity label creation
* integer label mapping
* minimal text preprocessing
* class weight computation
* stratified 80/10/10 train/validation/test split

## Project structure

* `pipeline/eda.py` – exploratory data analysis
* `pipeline/preprocess.py` – preprocessing pipeline
* `src/data/dataset.py` – dataset loader
* `data/processed/` – processed datasets
* `reports/` – EDA reports and visualizations

## Output files

* `train.parquet`
* `val.parquet`
* `test.parquet`
* `comment_level_dataset.parquet`
* `label_map.json`
* `class_weights.json`

## My role (Data Engineer)

Responsibilities completed:

* dataset loading and exploration
* EDA and target analysis
* annotation aggregation
* preprocessing pipeline
* severity label generation
* class weight computation
* train/validation/test split
* dataset packaging for model training

The processed dataset is ready for mBERT fine-tuning using the `text` column as input and the `label` column as the target variable.
