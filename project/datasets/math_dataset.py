import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from datasets import load_dataset
import os
from pathlib import Path
import json
import re

def prepare_data(max_rows=None, debug_mode=False):
    dataset = load_dataset("hendrycks/competition_math")

    if max_rows:
        dataset['train'] = dataset['train'].select(range(max_rows))
    elif debug_mode:
        dataset['train'] = dataset['train'].select(range(10))

    return dataset


def extract_level(text):
    match = re.search(r'\d+', text)
    if match:
        return int(match.group(0))
    else:
        return 0


class MathDataset(Dataset):
    def __init__(self, dir="MATH", mode="train", subset="", min_level=5):
        self.dir = Path(f"{dir}/{mode}")
        if subset:
            self.dir = Path(f"{dir}/{mode}/{subset}")

        self.problems = []
        for file in self.dir.rglob('*'):
            if file.is_file():  # Only consider files
                with open(file, 'r') as f:
                    raw_data = json.load(f)
                    if extract_level(raw_data["level"]) >= min_level:
                        content = {"problem" : raw_data["problem"], "level": raw_data["level"], "type": raw_data["type"]}, raw_data["solution"]
                        self.problems.append(content)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx]

if __name__ == '__main__':
    # data = prepare_data(debug_mode=True)

    # dataset = MathDataset()
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # for i, (data, label) in enumerate(dataloader):
    #     print(data["problem"][0])
    #     print(label[0])
    #     break
    # for i, (data, label) in enumerate(dataloader):
    #     print(data["problem"][0])
    #     print(label[0])
    #     break

    for i in range(10):
        print("min level: ", i)
        dataset = MathDataset(min_level=i)
        print(len(dataset))
        print(dataset[0])


"""
min level: 0; length 7500
min level: 1; length 7498
min level: 2; length 6934
min level: 3; length 5586
min level: 4; length 3994
min level: 5; length 2304
"""