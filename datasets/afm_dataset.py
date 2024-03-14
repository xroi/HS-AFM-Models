import numpy as np
import pickle


class AfmDataset:

    def __init__(self, data_path, pickle_amount):
        # data loading
        self.data_path = data_path
        self.pickle_amount = pickle_amount
        self.good_pickle_index_list = self._generate_good_pickle_index_list()
        self.good_pickle_amount = len(self.good_pickle_index_list)
        self.cur_pickle_i = self.good_pickle_index_list[0]
        self.cur_pickle = self._load_pickle_i(self.cur_pickle_i)
        self.single_data_raster_length = len(self.cur_pickle["rasterized_maps"])
        self.size_x = self.cur_pickle["rasterized_maps"][0].shape[0]
        self.size_y = self.cur_pickle["rasterized_maps"][0].shape[1]

    def _get_non_raster_i(self, raster_i, y_i, x_i):
        """This function assumes that each non rasterized image corresponds to a single pixel in a rasterized image"""
        dims = (self.single_data_raster_length, 2 * self.size_y, self.size_x)
        # *2 on y to account for return time
        return np.ravel_multi_index((raster_i, 2 * y_i, x_i), dims)

    @staticmethod
    def _load_pickle(file_name):
        """returns a dictionary with the following keys: real_time_maps, rasterized_maps, args"""
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    def _load_pickle_i(self, i):
        return self._load_pickle(f"{self.data_path}/{i}.pickle")

    def _generate_good_pickle_index_list(self):
        pickle_index_list = []
        i = 0
        for j in range(1, self.pickle_amount + 1):
            try:
                self._load_pickle(f"{self.data_path}/{j}.pickle")
                pickle_index_list.append(j)
                i += 1
            except Exception as _:
                pass
        return pickle_index_list
