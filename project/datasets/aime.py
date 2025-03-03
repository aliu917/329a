import os
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
    def __init__(self, file_path="aime_dataset_1.csv"):
        """
        Args:
            file_path (str): Path to the AIME dataset CSV file.
        """
        self.file_path = os.path.join(os.path.dirname(__file__), file_path)
        self.data = pd.read_csv(self.file_path)
        
        # Convert to list of tuples (problem, solution, answer)
        self.problems = list(zip(self.data["problem"], self.data["solution"], self.data["answer"]))

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem, solution, answer = self.problems[idx]
        return {"problem": [problem], "solution": [solution]}, str(answer)  # Ensure structure matches expected format

if __name__ == "__main__":
    dataset = AIMEDataset()
    print(f"Total problems: {len(dataset)}")
    
    # Display first problem
    data, labels = dataset[0]
    print("Data:", data)
    print("Labels:", labels)
