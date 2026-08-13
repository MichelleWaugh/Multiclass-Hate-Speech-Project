from datasets import Dataset
import pandas as pd
from pathlib import Path

class HateSpeechDataset:
    def __init__(self, data_dir="data/processed"):
        self.data_dir = Path(data_dir)

    def load_split(self, split):
        file_path = self.data_dir / f"{split}.parquet"
        df = pd.read_parquet(file_path)
        return Dataset.from_pandas(df, preserve_index=False)

    def load_all(self):
        return {
            "train": self.load_split("train"),
            "validation": self.load_split("val"),
            "test": self.load_split("test")
        }

if __name__ == "__main__":
    dataset = HateSpeechDataset()

    splits = dataset.load_all()

    print(splits)

    print("\nTrain sample:")
    print(splits["train"][0])