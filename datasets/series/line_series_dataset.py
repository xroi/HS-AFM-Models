import numpy as np
from datasets.afm_dataset import AfmDataset
from functools import reduce


class LineSeriesDataset(AfmDataset):
    def __init__(self, data_path, pickle_amount, line_y=20, get_back_frames=True):
        """ X = A series of full non rasterized HS-AFM images, corresponding to a single rasterized image.
        Y = A single rasterized image.
        """
        super().__init__(data_path, pickle_amount)
        self.get_back_frames = get_back_frames
        self.single_data_raster_length = self.single_data_raster_length * self.size_y
        self.n_samples = self.single_data_raster_length * self.good_pickle_amount
        self.line_y = line_y
        self.stacked_data = None

    def __getitem__(self, index):
        """x: 40/80 non rasterized frames (of forward and potentially back movement)
           y: the forward rastering line obtained from the first 40 frames of x"""
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        # dataset[i]
        pickle_i, raster_i = np.unravel_index(index, (self.good_pickle_amount,
                                                      self.single_data_raster_length))
        pickle_i = self.good_pickle_index_list[pickle_i]
        if pickle_i != self.cur_pickle_i:
            self.cur_pickle = self._load_pickle_i(pickle_i)
            self.cur_pickle_i = pickle_i

        frames_in_single_raster = self.size_y

        x = self.cur_pickle["non_rasterized_maps"][
            frames_in_single_raster * (raster_i * 2): frames_in_single_raster * (
                    (raster_i * 2) + (2 if self.get_back_frames else 1))]
        y = self._simple_line_raster(x, self.line_y)
        return x, y

    def __len__(self):
        # len(dataset)
        return self.n_samples

    def _simple_line_raster(self, maps, y):
        """This rastering assumes that the speed of the tip matches the non-rasterized frames. Also, it accounts for
         a single rasterized images"""
        raster = np.zeros(shape=(self.size_x, 1))
        for x in range(self.size_x):
            raster[x, 0] = maps[x][x, y]
        return raster

    def get_single_stacked_lines(self, i):
        stacked_lines = np.zeros(shape=(40, 1))
        x, _ = self.__getitem__(i)
        for j in range(40):
            cur = x[j][:, self.line_y: + self.line_y + 1]
            stacked_lines = np.hstack((stacked_lines, cur))
        stacked_lines = stacked_lines[:, 1:]
        return stacked_lines

    def get_center_of_mass(self, i, radius_around_center):
        # calculate temporal mean
        x, _ = self.__getitem__(i)
        x = np.dstack(x).mean(axis=2)

        # set values to minimum
        x = x - 47
        x[x < 0] = 0

        # set to 0 values outside of circular mask
        indices = np.indices(x.shape)
        mask = (indices[0] - 20) ** 2 + (indices[1] - 20) ** 2 > radius_around_center ** 2
        x[mask] = 0

        # calculate center of mass
        total = x.sum()
        x_coord = (x.sum(axis=1) @ range(x.shape[0])) / total
        y_coord = (x.sum(axis=0) @ range(x.shape[1])) / total
        return x_coord, y_coord

    def get_stacked_data(self):
        """
        :return: a 40x40xn array with non rasterized line scans. The diagonal entries form the rasterized line scan.
        """
        if self.stacked_data is not None:
            return self.stacked_data
        if self.get_back_frames is True:
            raise Exception("Cannot get regression formatted data if get_back_frames is True")
        res = np.zeros(shape=(40, 40, 1))
        for i in range(self.__len__()):
            stacked_lines = self.get_single_stacked_lines(i)[:, :, np.newaxis]
            res = np.concatenate((res, stacked_lines), axis=2)
        res = res[:, :, 1:]
        self.stacked_data = res
        return res

    def get_regression_formatted_data_(self, stacked_data):
        X_mat = stacked_data[0, 0, :][:, np.newaxis]
        for i in range(1, 40):
            X_mat = np.concatenate((X_mat, stacked_data[i, i, :][:, np.newaxis]), axis=1)
        return X_mat

    def get_regression2_formatted_data_(self, line_regression1):
        X_mat = line_regression1.pred(self.__getitem__(0)[1].T).flatten()[np.newaxis, :]
        for i in range(1, self.__len__()):
            X_mat = np.concatenate((X_mat, line_regression1.pred(self.__getitem__(i)[1].T).flatten()[np.newaxis, :]),
                                   axis=0)
        return X_mat

    # @staticmethod
    # def get_raster_x_y_by_index(i, size_x, size_y):
    #     if i % (size_x * 2) >= size_x:
    #         return None, None
    #     else:
    #         greater = False
    #         if i >= size_x * size_y:
    #             i -= size_x * size_y
    #             greater = True
    #         y, x = np.unravel_index(i, (size_x, size_y))
    #         if greater:
    #             y += size_y
    #         return x, int(y / 2)
