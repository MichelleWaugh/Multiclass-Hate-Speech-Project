import pandas as pd

train_path = "data/processed/train.parquet"

df = pd.read_parquet(train_path)

print("Columns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nLabel distribution:")
print(df["label"].value_counts())