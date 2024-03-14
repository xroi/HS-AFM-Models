import numpy as np
from datasets.afm_dataset import AfmDataset


class SingleTrajectoryDataset(AfmDataset):
    """X = a pixel and the adjacent ones on the x-axis
       Y = A single vector corresponding to the movement of the center of mass"""

    def __init__(self, data_path, pickle_amount, window_size):
        super().__init__(data_path, pickle_amount)
        if window_size % 2 == 0:
            raise ValueError("window_size should be odd")
        self.window_size = window_size
        self.buffer_size = int(window_size / 2)

        self.n_samples_per_raster = (self.size_y - self.BUFFER_SIZE * 2) * (self.size_x - self.BUFFER_SIZE * 2)
        self.n_samples_per_pickle = self.single_data_raster_length * self.n_samples_per_raster
        self.n_samples = self.good_pickle_amount * self.n_samples_per_pickle

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
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

        x_i = x_i + self.BUFFER_SIZE
        y_i = y_i + self.BUFFER_SIZE

        # x is of shape window_size + 2 x 1 from rasterized, final 2 are reserved for coordinates todo
        x = self.cur_pickle["rasterized_maps"][raster_i][x_i:x_i + self.WINDOW_SIZE, y_i + self.BUFFER_SIZE]
        y = self._get_single_trajectory_vector(raster_i, x_i, y_i)

        return x, y

    def __len__(self):
        # len(dataset)
        return self.n_samples

    def _get_single_trajectory_vector(self, raster_i, x_i, y_i):
        # Taking the average of the change of the center of mass in a window around the current pixel over
        # WINDOW_SIZE times
        center_of_masses = []
        for i in range(-self.buffer_size, self.buffer_size + 1):
            non_raster = self.cur_pickle["real_time_maps"][
                self._get_non_raster_i(raster_i, y_i, x_i + i)]
            non_raster = non_raster[x_i - self.buffer_size:x_i + self.buffer_size + 1,
                         y_i - self.buffer_size:y_i + self.buffer_size + 1]
            center_of_masses.append(self._get_center_of_mass(non_raster))
        mass_change_vectors = []
        for i in range(len(center_of_masses) - 1):
            mass_change_vectors.append(center_of_masses[i + 1] - center_of_masses[i])
        mass_change_vectors = np.array(mass_change_vectors)
        return np.mean(mass_change_vectors, axis=0)

    def _get_center_of_mass(self, arr):
        # Format the array as [[x,y,weight],...]
        masses = []
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                masses.append([x - self.BUFFER_SIZE, y - self.BUFFER_SIZE, arr[x, y]])
        masses = np.array(masses)
        # return [x,y] of center of mass
        return np.average(masses[:, :2], axis=0, weights=masses[:, 2])
