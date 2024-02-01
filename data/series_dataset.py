import numpy as np
from afm_dataset import AfmDataset


class TrajectoryDataset(AfmDataset):
    def __init__(self, data_path, pickle_amount):
        super().__init__(data_path, pickle_amount)

        self.n_samples = self.single_data_raster_length * self.good_pickle_amount

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        # dataset[i]
        pickle_i, raster_i = np.unravel_index(index, (self.good_pickle_amount,
                                                      self.single_data_raster_length))
        pickle_i = self.good_pickle_index_list[pickle_i]
        if pickle_i != self.cur_pickle_i:
            self.cur_pickle = self._load_pickle_i(pickle_i)
            self.cur_pickle_i = pickle_i

        # (assuming output interval is equal to tip movement interval)
        # (*2 is to account for return time)
        frames_in_single_raster = self.size_x * self.size_y * 2

        x = self.cur_pickle["real_time_maps"][
            frames_in_single_raster * raster_i: frames_in_single_raster * (raster_i + 1)]
        y = self.cur_pickle["rasterized_maps"][raster_i]
        return x, y

    def __len__(self):
        # len(dataset)
        return self.n_samples
