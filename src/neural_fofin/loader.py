import json
import os

from torch.utils.data import Dataset, random_split


class JsonDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.files = os.listdir(directory)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def test_train_split(self, train_size):
        train_size = int(train_size * len(self))
        test_size = len(self) - train_size
        return random_split(self, [train_size, test_size])
