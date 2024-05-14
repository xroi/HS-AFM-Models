from itertools import product

import numpy as np


class MovementSeries:
    def __init__(self, start_i):
        """
        :param start_i: the index in the images series in which the object appears
        positions: the position (x,y) of the center of the object at every image in the until it disappears
        sizes: the size (total amount of pixels decreed to be a part of the object being tracked) at each point of time
        """
        self._start_i = start_i
        self._positions = []
        # self._sizes = []

    def __len__(self):
        return len(self._positions)

    def get_start_i(self):
        return self._start_i

    def get_final_i(self):
        return self._start_i + self.get_length()

    def get_length(self):
        return len(self._positions)

    def get_positions(self):
        return self._positions

    def get_ith_position(self, i):
        return self._positions[i]

    def get_ith_movement(self, i):
        pos1 = self.get_ith_position(i - 1)
        pos2 = self.get_ith_position(i)
        return pos2[0] - pos1[0], pos2[1] - pos1[1]

    def get_ith_angle(self, i):
        x, y = self.get_ith_movement(i)
        if x == 0 and y == 0:
            return None
        inv = np.arctan2(y, x)
        degree = np.mod(np.degrees(inv), 360)
        return degree

    def get_ith_direction_4(self, i):
        degree = self.get_ith_angle(i)
        if degree is None:
            return -1
        if 0 <= degree < 90:
            return 0
        elif 90 <= degree < 180:
            return 1
        elif 180 <= degree < 270:
            return 2
        else:
            return 3

    def get_ith_direction_8(self, i):
        degree = self.get_ith_angle(i)
        if degree is None:
            return -1
        if 0 <= degree < 45:
            return 0
        elif 45 <= degree < 90:
            return 1
        elif 90 <= degree < 135:
            return 2
        elif 135 <= degree < 180:
            return 3
        elif 180 <= degree < 225:
            return 4
        elif 225 <= degree < 270:
            return 5
        elif 270 <= degree < 315:
            return 6
        else:
            return 7

    def get_final_position(self):
        return self._positions[-1]

    def add_position(self, x, y):
        self._positions.append((x, y))

    def is_i_in_range(self, i):
        if self.get_start_i() <= i < self.get_final_i():
            return True
        return False


class MovementTracker:
    def __init__(self, image_series, window_r, include_r):
        """
        :param image_series: Series of non rasterized afm images.
        :param window_r: The radius (in px) used for tracking (finding the local max in window_r around each point).
        :param include_r: The radius (in px) from the center of the images in which local maxima are considered.
        """
        self._image_series = image_series
        self._window_r = window_r
        self._include_radius = include_r

    def get_movement_series_list(self):
        movement_series_list = []
        for im_i, im in enumerate(self._image_series):
            self.add_frame_to_series_array(movement_series_list, im, im_i)
        return movement_series_list

    def add_frame_to_series_array(self, movement_series_array, im, im_i):
        belong_i = -np.ones(shape=im.shape)
        # belong_i holds, for every pixel, what is the index of the movement series to which it belongs
        for x, y in product(range(im.shape[0]), range(im.shape[1])):
            if belong_i[x, y] != -1:
                continue
            if (x - 20) ** 2 + (y - 20) ** 2 > 18.5 ** 2:
                belong_i[x, y] = -2
                continue
            self.iterate_to_find_max(im, x, y, belong_i, movement_series_array, im_i)

    def iterate_to_find_max(self, im, x, y, belong_i, movement_series_array, im_i):
        max_coord, mask = self.max_in_radius(im, x, y, self._window_r)
        i = belong_i[max_coord[0], max_coord[1]]
        if i != -1:
            # existing maximum point
            belong_i[mask] = i
            return i
        if im[max_coord[0], max_coord[1]] == im[x, y]:
            # new maximum point
            i = self.add_to_movement_series_array(x, y, movement_series_array, im_i)
            belong_i[mask] = i
            return i
        i = self.iterate_to_find_max(im, max_coord[0], max_coord[1], belong_i,
                                     movement_series_array, im_i)
        belong_i[mask] = i
        return i

    @staticmethod
    def max_in_radius(arr, x, y, r):
        # Create an array of indices
        indices = np.indices(arr.shape)
        # Create a circular mask
        mask = (indices[0] - x) ** 2 + (indices[1] - y) ** 2 <= r ** 2
        # Apply the mask to the array
        masked_arr = np.where(mask, arr, np.nan)
        # Find the index of the maximum value in the masked array
        max_coord = np.unravel_index(np.nanargmax(masked_arr), arr.shape)
        return max_coord, mask

    def add_to_movement_series_array(self, x, y, movement_series_array, im_i):
        # ignore local maxima outside of include_radius
        if np.sqrt((x - 20) ** 2 + (y - 20) ** 2) > self._include_radius:
            return -2
        for i, series in enumerate(movement_series_array):
            if series.get_final_i() != im_i:
                continue
            old_x, old_y = series.get_final_position()
            if (old_x - x) ** 2 + (old_y - y) ** 2 <= self._window_r ** 2:
                series.add_position(x, y)
                return i
        series = MovementSeries(im_i)
        series.add_position(x, y)
        movement_series_array.append(series)
        return len(movement_series_array) - 1
