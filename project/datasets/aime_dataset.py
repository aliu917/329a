import os
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd


def prepare_data(file_path, max_rows=None, debug_mode=False):
    """Loads and optionally filters the AIME dataset."""
    data = pd.read_csv(file_path)
    
    if max_rows:
        data = data.head(max_rows)
    elif debug_mode:
        data = data.head(10)
    
    return data


class AIMEDataset(Dataset):
    def __init__(self, dir="AIME", file_path="aime_dataset_1.csv", min_level=0):
        """
        Args:
            dir (str): Directory containing the dataset
            file_path (str): Path to the AIME dataset CSV file
            min_level (int): Minimum difficulty level (not used for AIME but kept for compatibility)
        """
        self.file_path = Path(f"{dir}/{file_path}")
        self.data = pd.read_csv(self.file_path)
        
        # Convert to the same format as MathDataset
        self.problems = []
        for _, row in self.data.iterrows():
            # Create content in the same format as MathDataset
            content = (
                {"problem": row["problem"], "level": "Level 5", "type": "AIME"}, 
                row["solution"]
            )
            self.problems.append(content)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx]


if __name__ == "__main__":
    dataset = AIMEDataset()
    print(f"Dataset length: {len(dataset)}")
    problem_data, solution = dataset[0]
    print("Problem metadata:", problem_data)
    print("Solution:", solution)