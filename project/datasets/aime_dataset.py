import os
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import re
# from sklearn.model_selection import train_test_split

def extract_first_solution(solution_text):
    """
    Extracts only the first solution from a text that may contain multiple solutions.
    
    Args:
        solution_text (str): The full solution text possibly containing multiple solutions
        
    Returns:
        str: Only the first solution
    """
    # Check if the solution contains multiple solutions denoted by "== Solution X =="
    if "== Solution 2 ==" in solution_text:
        # Split by the solution marker and take only the first part
        first_solution = solution_text.split("== Solution 2 ==")[0].strip()
        return first_solution
    
    # If there's no explicit "== Solution 2 ==" marker but there are other solution markers
    solution_markers = re.findall(r"==\s*Solution\s+\d+\s*==", solution_text)
    if solution_markers:
        # Split by the first occurrence of any solution marker
        pattern = r"==\s*Solution\s+\d+\s*=="
        parts = re.split(pattern, solution_text, maxsplit=1)
        if len(parts) > 1:
            # If the first part is very short, it might not be a solution but just an intro
            # In that case, take the text up to the second solution marker
            if len(parts[0].strip()) < 50 and len(solution_markers) > 1:
                # Find the position of the second solution marker
                second_marker_pos = solution_text.find(solution_markers[1])
                if second_marker_pos > -1:
                    return solution_text[:second_marker_pos].strip()
            else:
                return parts[0].strip()
    
    # If no solution markers found, return the original text
    return solution_text


class AIMEDataset(Dataset):
    def __init__(self, dir="AIME", file_path="aime_dataset_1.csv", min_level=0, parse_solutions=True, mode="train"):
        """
        Args:
            dir (str): Directory containing the dataset
            file_path (str): Path to the AIME dataset CSV file
            min_level (int): Minimum difficulty level (not used for AIME but kept for compatibility)
            parse_solutions (bool): Whether to extract only the first solution from multi-solution texts
        """
        self.file_path = Path(f"{dir}/{file_path}")
        data = pd.read_csv(self.file_path)
        # self.data, val_df = train_test_split(data, test_size=0.2, random_state=42)
        data.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(0.8 * len(data))
        if mode == "train":
            self.data = data[:split]
        else:
            self.data = data[split:]

        self.parse_solutions = parse_solutions
        
        # Convert to the same format as MathDataset
        self.problems = []
        for _, row in self.data.iterrows():
            solution = row["solution"]
            
            # If enabled, extract only the first solution
            if parse_solutions:
                solution = extract_first_solution(solution)
            
            # Create content in the same format as MathDataset
            content = (
                {"problem": row["problem"], "level": "Level 5", "type": "AIME", "answer": row["answer"]}, 
                solution
            )
            self.problems.append(content)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx]


if __name__ == "__main__":
    # Example usage
    dataset = AIMEDataset(parse_solutions=True)
    print(f"Dataset length: {len(dataset)}")
    
    # Print an example to show the parsing
    problem_data, solution = dataset[0]
    print("Problem:", problem_data["problem"], "...")
    print("\nAnswer:", problem_data["answer"])
    print("\nSolution (first only):", solution, "...")
    
    # Show comparison with unparsed solutions
    dataset_full = AIMEDataset(parse_solutions=False)
    _, full_solution = dataset_full[0]
    print(f"\nFull solution length: {len(full_solution)} chars")
    print(f"Parsed solution length: {len(solution)} chars")