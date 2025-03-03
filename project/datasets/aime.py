from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


class AIMEDataset(Dataset):
    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the AIME dataset CSV file.
        """
        self.file_path = Path(file_path)
        self.data = pd.read_csv(self.file_path)
        
        # Convert to list of tuples (problem, solution, answer)
        self.problems = list(zip(self.data["problem"], self.data["solution"], self.data["answer"]))

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx]


if __name__ == "__main__":
    dataset = AIMEDataset("aime_dataset_1.csv")
    print(f"Total problems: {len(dataset)}")
    
    # Display first problem
    problem, solution, answer = dataset[0]
    print("Problem:", problem)
    print("Solution:", solution)
    print("Answer:", answer)

