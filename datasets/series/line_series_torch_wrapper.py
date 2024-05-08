import numpy as np
from torch.utils.data import Dataset
import torch


class LineSeriesTorchWrapper(Dataset):
    def __init__(self, dataset, regression_model):
        self.dataset = dataset
        self.regression_model = regression_model

    def __getitem__(self, index):
        _, line = self.dataset[index]
        x = self.regression_model.pred(line.T)
        y = self.dataset.get_single_stacked_lines(index)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.dataset)
