import numpy as np
import scipy.stats
from datasets.series.series_dataset import SeriesDataset


class SlidingMeanDataset(SeriesDataset):
    def __init__(self, data_path, pickle_amount, temporal_window_size=0, spatial_window_size=0, use_original_y=False,
                 discard_start_end=False):
        super().__init__(data_path, pickle_amount)
        self.temporal_window_size = temporal_window_size
        self.spacial_window_size = spatial_window_size
        self.use_original_y = use_original_y
        self.discard_start_end = discard_start_end
        self.cached_i = -1
        self.cached_x, self.cached_y, self.cached_orig_x = None, None, None

    def __getitem__(self, index):
        if index == self.cached_i:
            return self.cached_x, self.cached_y, self.cached_orig_x
        x, y = SeriesDataset.__getitem__(self, index)
        orig_x = x
        if self.spacial_window_size > 0:
            x = [self._gaussian_filter(im, self.spacial_window_size) for im in x]
        if self.temporal_window_size > 0:
            x = np.dstack(x)
            x = np.apply_along_axis(self._moving_average, 2, x, self.temporal_window_size)
            x = [x[:, :, i] for i in range(x.shape[2])]
        if self.discard_start_end:
            x = x[self.temporal_window_size:len(x) - self.temporal_window_size]
            orig_x = orig_x[self.temporal_window_size:len(x) - self.temporal_window_size]
        if not self.use_original_y:
            y = self._simple_raster(x)
        self.cached_x = x
        self.cached_y = y
        self.cached_orig_x = orig_x
        self.cached_i = index
        return x, y, orig_x

    @staticmethod
    def _moving_average(x, w):
        return np.convolve(x, np.ones(w), mode='same') / w

    @staticmethod
    def _gaussian_filter(x, w):
        return scipy.ndimage.gaussian_filter(input=x, sigma=w, radius=w, axes=(0, 1))
