import numpy as np
from datasets.series.line_series_dataset import LineSeriesDataset
from torch.utils.data import Dataset
import torch


class LineSeriesTorchWrapper(Dataset):
    def __init__(self, dataset: LineSeriesDataset, com_radius_around_center, max_seq_len):
        self.dataset = dataset
        self.com_radius_around_center = com_radius_around_center
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        x = np.zeros(shape=(1, 40))  # len(seq) x n_features
        y = np.zeros(shape=(1, 3))  # len(seq) x n_targets
        for i in range(self.max_seq_len):
            _, line = self.dataset[index * self.max_seq_len + i]
            if self.dataset.is_new_seq and i != 0:
                break
            x = np.concatenate((x, line.T), axis=0)
            # y1, y2 = self.dataset.get_center_of_mass(index * self.max_seq_len + i, self.com_radius_around_center)
            y1 = self.dataset.get_local_maxima(index * self.max_seq_len + i,
                                               self.com_radius_around_center)
            y = np.concatenate((y, y1[np.newaxis, :]))
        x = x[1:, :]
        y = y[1:, :]
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return int(len(self.dataset) / self.max_seq_len) - 1

    def get_matching_non_rasters(self, index):
        x = []
        for i in range(self.max_seq_len):
            non_raster, _ = self.dataset[index * self.max_seq_len + i]
            if self.dataset.is_new_seq and i != 0:
                break
            x.append(np.mean(np.array(non_raster), axis=0))
        return x
