import numpy as np
from torch.utils.data import Dataset
import torch


class LineSeriesTorchWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        y, x = self.dataset[index]
        stacked_lines = np.zeros(shape=(40, 1))
        for j in range(40):
            x = y[j][:, self.dataset.line_y: + self.dataset.line_y + 1]
            stacked_lines = np.hstack((stacked_lines, x))
        stacked_lines = stacked_lines[:, 1:]

        # stacked_x = np.zeros(shape=(40, 1))
        # for j in range(40):
        #     stacked_x = np.hstack((stacked_x, [x[j]] * 40))
        # stacked_x = stacked_x[:, 1:]

        x = torch.from_numpy(x)
        y = torch.from_numpy(stacked_lines)

        # min max normalize x and y to be between 0 and 1
        cur_min, cur_max = 40, 60
        new_min, new_max = 0, 1
        x = (x - cur_min) / (cur_max - cur_min) * (new_max - new_min) + new_min
        y = (y - cur_min) / (cur_max - cur_min) * (new_max - new_min) + new_min

        return x, y

    def __len__(self):
        return len(self.dataset)
