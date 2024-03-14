import numpy as np
from datasets.afm_dataset import AfmDataset
from datasets.series.sliding_mean_dataset import SlidingMeanDataset
from models.statistical.movment_tracker import MovementTracker, MovementSeries


class TrackedDataset:
    """
    Let a single horizontal line in the image be of length n.
    X = A n+4 vector, holding a single line of a rasterized image, the coordinate of the tip along the line,
    and the location of the closest tracked point to the tip in the previous frame.
    Y = A 2 vector with the x,y locations of the current location of the closest point. If it is too fast, holds (-1,-1)
    """

    def __init__(self, sliding_mean_dataset: SlidingMeanDataset):
        self.n_samples = (sliding_mean_dataset.n_samples * sliding_mean_dataset.size_x * sliding_mean_dataset.size_y)
        # - sliding_mean_dataset.good_pickle_amount * sliding_mean_dataset.single_data_raster_length)
        self.sliding_mean_dataset = sliding_mean_dataset
        self.cur_sliding_i = -1
        self.prev_y = np.array([-1, -1])

    def __getitem__(self, index):
        # dataset[i]
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        pickle_i, raster_i, y_i, x_i = np.unravel_index(index,
                                                        (self.sliding_mean_dataset.good_pickle_amount,
                                                         self.sliding_mean_dataset.single_data_raster_length,
                                                         self.sliding_mean_dataset.size_y,
                                                         self.sliding_mean_dataset.size_x))
        sliding_i = pickle_i * self.sliding_mean_dataset.single_data_raster_length + raster_i
        if sliding_i != self.cur_sliding_i:
            self.cur_sliding_i = sliding_i
            full_x, self.full_y = self.sliding_mean_dataset[sliding_i]
            movement_tracker = MovementTracker(full_x, 4)
            self.movement_series_list = movement_tracker.get_movement_series_list()
        # x_i += sliding_i

        x = self.full_y[:, y_i].tolist() + [self.prev_y[0], self.prev_y[1]]
        closest_x, closest_y = self.get_closest_tracked_point(x_i, y_i)
        y = np.array([closest_x, closest_y])
        self.prev_y = y

        return x, y

    def __len__(self):
        # len(dataset)
        return self.n_samples

    def get_closest_tracked_point(self, x_i, y_i):
        i = x_i + (y_i * self.sliding_mean_dataset.size_x * 2)
        top_x, top_y = -np.inf, -np.inf
        for series in self.movement_series_list:
            if series.is_i_in_range(i):
                x, y = series.get_ith_position(i - series.get_start_i())
                if (x_i - x) ** 2 + (y_i - y) ** 2 > (x_i - top_x) ** 2 + (y_i - top_y) ** 2:
                    top_x = x
                    top_y = y
        return top_x, top_y
