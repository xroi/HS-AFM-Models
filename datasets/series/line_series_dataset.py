import numpy as np
from datasets.afm_dataset import AfmDataset


class LineSeriesDataset(AfmDataset):
    def __init__(self, data_path, pickle_amount, line_y=20):
        """ X = A series of full non rasterized HS-AFM images, corresponding to a single rasterized image.
        Y = A single rasterized image.
        """
        super().__init__(data_path, pickle_amount)
        self.single_data_raster_length = self.single_data_raster_length * self.size_y
        self.n_samples = self.single_data_raster_length * self.good_pickle_amount
        self.line_y = line_y

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

        frames_in_single_raster = self.size_y

        x = self.cur_pickle["non_rasterized_maps"][
            frames_in_single_raster * (raster_i * 2): frames_in_single_raster * ((raster_i * 2) + 2)]
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
