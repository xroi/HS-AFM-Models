from torch.utils.data import Dataset
import torch


class TorchWrapperDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.dataset)
