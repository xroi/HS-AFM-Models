import numpy as np
from data.afm_dataset import AfmDataset
from itertools import product


class VectorFieldTrajectoryDataset(AfmDataset):
    """X = a rasterized image
       Y = A vector field corresponding to the movement of the center of mass"""

    def __init__(self, data_path, pickle_amount, window_size):
        super().__init__(data_path, pickle_amount)
        if window_size % 2 == 0:
            raise ValueError("window_size should be odd")
        self.window_size = window_size
        self.buffer_size = int(window_size / 2)

        self.n_samples = self.good_pickle_amount * self.single_data_raster_length

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

        x = self.cur_pickle["rasterized_maps"][raster_i]

        y = np.zeros(shape=(x.shape[0], x.shape[1], 2))
        for x_i, y_i in product(range(self.size_x), range(self.size_y)):
            if (not (self.buffer_size <= x_i < self.size_x - self.buffer_size)) or \
                    (not (self.buffer_size <= y_i < self.size_y - self.buffer_size)):
                y[x_i, y_i, :] = np.array([0, 0])
            else:
                y[x_i, y_i, :] = self._get_single_trajectory_vector(raster_i, x_i, y_i)
        return x, y

    def _get_single_trajectory_vector(self, raster_i, x_i, y_i):
        # Taking the average of the change of the center of mass in a window around the current pixel over
        # WINDOW_SIZE times
        center_of_masses = []
        for i in range(-self.buffer_size, self.buffer_size + 1):
            # for i in [0]:
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

    def __len__(self):
        # len(dataset)
        return self.n_samples

    def _get_center_of_mass(self, arr):
        # Format the array as [[x,y,weight],...]
        masses = []
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                masses.append([x - self.buffer_size, y - self.buffer_size, arr[x, y]])
        masses = np.array(masses)
        # return [x,y] of center of mass
        return np.average(masses[:, :2], axis=0, weights=masses[:, 2])
