import numpy as np
from afm_dataset import AfmDataset


class TrajectoryDataset(AfmDataset):
    WINDOW_SIZE = 5  # Should be odd
    BUFFER_SIZE = int(WINDOW_SIZE / 2)

    def __init__(self, data_path, pickle_amount):
        super().__init__(data_path, pickle_amount)

        self.n_samples_per_raster = (self.size_y - self.BUFFER_SIZE * 2) * (self.size_x - self.BUFFER_SIZE * 2)
        self.n_samples_per_pickle = self.single_data_raster_length * self.n_samples_per_raster
        self.n_samples = self.good_pickle_amount * self.n_samples_per_pickle

    def __getitem__(self, index):
        # dataset[i]
        pickle_i, raster_i, y_i, x_i = np.unravel_index(index,
                                                        (self.good_pickle_amount,
                                                         self.single_data_raster_length,
                                                         self.size_y - self.BUFFER_SIZE * 2,
                                                         self.size_x - self.BUFFER_SIZE * 2))
        pickle_i = self.good_pickle_index_list[pickle_i]
        if pickle_i != self.cur_pickle_i:
            self.cur_pickle = self._load_pickle_i(pickle_i)
            self.cur_pickle_i = pickle_i

        # x is of shape window_size + 2 x 1 from rasterized, final 2 are reserved for coordinates todo
        x = self.cur_pickle["rasterized_maps"][raster_i][x_i:x_i + self.WINDOW_SIZE, y_i + self.BUFFER_SIZE]

        # Taking the average of the change of the center of mass in a window around the current pixel over
        # WINDOW_SIZE times
        center_of_masses = []
        for i in range(self.WINDOW_SIZE):
            non_raster = self.cur_pickle["real_time_maps"][
                self._get_non_raster_i(raster_i, y_i, x_i + i)]
            non_raster = non_raster[x_i:x_i + self.WINDOW_SIZE, y_i:y_i + self.WINDOW_SIZE]
            center_of_masses.append(self._get_center_of_mass(non_raster))
        mass_change_vectors = []
        for i in range(len(center_of_masses) - 1):
            mass_change_vectors.append(center_of_masses[i + 1] - center_of_masses[i])
        mass_change_vectors = np.array(mass_change_vectors)
        y = np.mean(mass_change_vectors, axis=0)
        return x, y

    def __len__(self):
        # len(dataset)
        return self.n_samples

    def _get_center_of_mass(self, arr):
        # Format the array as [[x,y,weight],...]
        masses = []
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                masses.append([x - self.BUFFER_SIZE, y - self.BUFFER_SIZE, arr[x, y]])
        masses = np.array(masses)
        # return [x,y] of center of mass
        return np.average(masses[:, :2], axis=0, weights=masses[:, 2])
