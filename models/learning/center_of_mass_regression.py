import sys

from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from datasets.series.line_series_dataset import LineSeriesDataset
import numpy as np
import pickle


class LineRegression1:
    def __init__(self):
        self.model = LinearRegression()
        self.RADIUS_AROUND_CENTER = 11
        self.SPATIAL_WINDOW_SIZE = 0
        self.TEMPORAL_WINDOW_SIZE = 0

    def train_regression_models(self):
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40", 100, line_y=20, get_back_frames=False,
                                    temporal_window_size=self.TEMPORAL_WINDOW_SIZE,
                                    spatial_window_size=self.SPATIAL_WINDOW_SIZE)
        # stacked_data = dataset.get_stacked_data()
        # regression_X = dataset.get_regression_formatted_data(stacked_data)  # n_samples x n_features

        regression_X, regression_y = self.format_regression_x_y(dataset)
        # poly = PolynomialFeatures(2, interaction_only=True)
        # regression_X = poly.fit_transform(regression_X)
        self.model.fit(regression_X, regression_y)

    def test_model(self):
        # pickles 75 to 100
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40/test", 100, line_y=20, get_back_frames=False,
                                    temporal_window_size=self.TEMPORAL_WINDOW_SIZE,
                                    spatial_window_size=self.SPATIAL_WINDOW_SIZE)
        regression_X, ground_truth = self.format_regression_x_y(dataset)
        pred_values = self.model.predict(regression_X)

        residuals = (ground_truth - pred_values) ** 2
        print(np.mean(residuals, axis=0))

        # np.set_printoptions(threshold=sys.maxsize)
        # import matplotlib.pyplot as plt
        # plt.scatter(pred_values[20, 23, :], residuals[20, 23, :])
        # plt.show()

    def format_regression_x_y(self, dataset):
        prev_pickle = -1
        prev_x_com, prev_y_com = None, None
        regression_X = np.zeros(shape=(1, 42))  # n_samples x n_features
        regression_y = np.zeros(shape=(1, 2))  # n_samples x n_targets
        for i in range(len(dataset)):
            _, y = dataset[i]
            if dataset.sliding_mean_dataset.cur_pickle != prev_pickle:
                # ignore the first entry in each pickle because it has no previous entry
                prev_pickle = dataset.sliding_mean_dataset.cur_pickle
                prev_x_com, prev_y_com = dataset.get_center_of_mass(i, self.RADIUS_AROUND_CENTER)
                continue
            x_com, y_com = dataset.get_center_of_mass(i, self.RADIUS_AROUND_CENTER)
            regression_X = np.concatenate((regression_X, np.concatenate((y, np.array([prev_x_com, prev_y_com])))))
            regression_y = np.concatenate((regression_y, np.array([x_com, y_com])))
            prev_x_com, prev_y_com = x_com, y_com
        regression_X = regression_X[1:, :]
        regression_y = regression_y[1:, :]
        return regression_X, regression_y

    def pred(self, x):
        """Given a rasterized vector, concatenated to [prev_com_x,prev_com_y] return predicted [cur_com_x,cur_com_y]"""
        return self.model.predict(x)

    def save_model(self, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_mode(self, file_name):
        with open(file_name, 'rb') as handle:
            self.model = pickle.load(handle)
