import numpy as np
from datasets.afm_dataset import AfmDataset


class SeriesDataset(AfmDataset):
    def __init__(self, data_path, pickle_amount):
        """ X = A series of full non rasterized HS-AFM images, corresponding to a single rasterized image.
        Y = A single rasterized image.
        """
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

        x = self.cur_pickle["non_rasterized_maps"][
            frames_in_single_raster * raster_i: frames_in_single_raster * (raster_i + 1)]
        y = self.cur_pickle["rasterized_maps"][raster_i]
        return x, y

    def __len__(self):
        # len(dataset)
        return self.n_samples

    def _simple_raster(self, maps):
        """This rastering assumes that the speed of the tip matches the non-rasterized frames. Also it accounts for
         a single rasterized images"""
        raster = np.zeros(shape=(self.size_x, self.size_y))
        x, y = 0, 0
        for i in range(len(maps)):
            if i % (self.size_x * 2) >= self.size_x:
                continue
            elif i % (self.size_x * 2) == self.size_x - 1:
                raster[x, y] = maps[i][x, y]
                y += 1
                x = 0
                continue
            raster[x, y] = maps[i][x, y]
            x += 1
        return raster

    @staticmethod
    def get_raster_x_y_by_index(i, size_x, size_y):
        if i % (size_x * 2) >= size_x:
            return None, None
        else:
            greater = False
            if i >= size_x * size_y:
                i -= size_x * size_y
                greater = True
            y, x = np.unravel_index(i, (size_x, size_y))
            if greater:
                y += size_y
            return x, int(y / 2)
