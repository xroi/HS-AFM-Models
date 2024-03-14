import numpy as np
from datasets.afm_dataset import AfmDataset


class FullCompLineMeanHeightDataset(AfmDataset):
    """X = a line from a rasterized image at radius r around the center
       Y = Mean height at a radius r from the center, along the frames {-r/2,...,r/2) around the center."""

    def __init__(self, data_path, pickle_amount, r, buffer):
        super().__init__(data_path, pickle_amount)
        self.n_samples = (self.size_y - 2 * buffer) * self.single_data_raster_length * self.good_pickle_amount
        self.r = r
        self.buffer = buffer

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        # dataset[i]
        pickle_i, raster_i, y_i = np.unravel_index(index, (self.good_pickle_amount,
                                                           self.single_data_raster_length,
                                                           (self.size_y - 2 * self.buffer)))
        pickle_i = self.good_pickle_index_list[pickle_i]
        if pickle_i != self.cur_pickle_i:
            self.cur_pickle = self._load_pickle_i(pickle_i)
            self.cur_pickle_i = pickle_i
        y_i += self.buffer

        x = self.cur_pickle["rasterized_maps"][raster_i][
            int((self.size_x / 2) - self.r):int((self.size_x / 2) + self.r), y_i]
        y_datas = [self.cur_pickle["real_time_maps"][self._get_non_raster_i(raster_i, y_i, x_i)] for x_i in
                   range(int((self.size_x / 2) - self.r), int((self.size_x / 2) + self.r))]
        y_means = [np.mean(data) for data in y_datas]
        y = np.mean(y_means)

        return x, y

    def __len__(self):
        # len(dataset)
        return self.n_samples

    @staticmethod
    def _get_entries_within_radius(data, r):
        center = np.array(data.shape) / 2
        y, x = np.ogrid[:data.shape[0], :data.shape[1]]
        dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        mask = dist_from_center < r
        return data[mask]
