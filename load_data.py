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
from utils import *

def prepare_data(max_rows=None, debug_mode=False):
    dataset = load_dataset("hendrycks/competition_math")

    if max_rows:
        dataset['train'] = dataset['train'].select(range(max_rows))
    elif debug_mode:
        dataset['train'] = dataset['train'].select(range(10))

    return dataset


class MathDataset(Dataset):
    def __init__(self, dir="MATH", mode="train", subset=""):
        self.dir = Path(f"{dir}/{mode}")
        if subset:
            self.dir = Path(f"{dir}/{mode}/{subset}")

        self.file_paths = []
        for file in self.dir.rglob('*'):
            if file.is_file():  # Only consider files
                self.file_paths.append(file)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with open(path, 'r') as f:
            raw_data = json.load(f)
            # answer = re.findall(r'boxed\{(.*?)\}', raw_data["solution"])[0]
            # answer = extract_text_within_box(raw_data["solution"])
            #TODO: decompose steps??
            return {"problem" : raw_data["problem"], "level": raw_data["level"], "type": raw_data["type"]}, raw_data["solution"]


if __name__ == '__main__':
    # data = prepare_data(debug_mode=True)
    dataset = MathDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i, (data, label) in enumerate(dataloader):
        print(data["problem"][0])
        print(label[0])
        break
    for i, (data, label) in enumerate(dataloader):
        print(data["problem"][0])
        print(label[0])
        break