# Processed dataset

This folder contains the processed version of the Hugging Face dataset `Shubhi324/measuring-hate-speech`.

## Objective

The original dataset contains multiple annotation rows for the same comment. The preprocessing pipeline converts the dataset into a comment-level dataset suitable for transformer-based multiclass hate speech classification.

## Preprocessing steps

1. Aggregated all annotation rows by `comment_id`
2. Preserved the first text instance for each comment
3. Averaged `hate_speech_score` across annotators
4. Preserved all `target_*` columns (53 target categories) using max aggregation
5. Created five severity classes:

   * not_hate
   * weak_hate
   * moderate_hate
   * strong_hate
   * extreme_hate
6. Created integer labels (0–4)
7. Applied minimal text preprocessing:

   * removed URLs
   * removed @mentions
   * normalized whitespace
8. Computed class weights for imbalanced classification
9. Performed a stratified train/validation/test split (80/10/10)

## Output files

* `comment_level_dataset.parquet` – complete processed dataset
* `train.parquet` – training split
* `val.parquet` – validation split
* `test.parquet` – test split
* `label_map.json` – integer label mapping
* `class_weights.json` – class weights for model training

## Final dataset

* Original rows: 135,556
* Comment-level rows: 39,565
* Target dimensions preserved: 53

The processed dataset is ready for mBERT fine-tuning using the `text` column as input and the `label` column as the target variable.
