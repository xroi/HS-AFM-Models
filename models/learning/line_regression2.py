import sys

from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lars, ElasticNet
from datasets.series.line_series_dataset import LineSeriesDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from models.learning.line_regression1 import LineRegression1
from scipy.ndimage import gaussian_filter
import numpy as np
import pickle


class LineRegression2:
    def __init__(self):
        lin_reg = ElasticNet(alpha=0.37)
        self.model = RegressorChain(lin_reg, verbose=True, order=LineRegression2.get_diagonal_indices(40, 39))

    def train_regression_models(self):
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40", 75, line_y=20, get_back_frames=False)
        stacked_data = dataset.get_stacked_data()
        regression_X = dataset.get_regression_formatted_data_(stacked_data)

        first = np.zeros(shape=(1, 1560))
        for i in range(stacked_data.shape[2]):
            temp = self.remove_diagonal(stacked_data[:, :, i]).flatten()[np.newaxis, :]
            first = np.concatenate((first, temp), axis=0)
        first = first[1:, :]
        self.model.fit(regression_X, first)

    def test_model(self):
        # pickles 75 to 100
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40/test", 100, line_y=20,
                                    get_back_frames=False)
        ground_truth = np.zeros(shape=(40, 40, len(dataset)))
        pred_values = np.zeros(shape=(40, 40, len(dataset)))
        residuals = np.zeros(shape=(40, 40, len(dataset)))
        for i in range(len(dataset)):
            x = dataset.get_single_stacked_lines(i)
            ground_truth[:, :, i] = x
            y = self.pred(dataset[i][1].T)
            pred_values[:, :, i] = y
            residuals[:, :, i] = (x - y) ** 2
            print(i)
        print(np.mean(residuals, axis=(0, 1, 2)))
        np.set_printoptions(threshold=sys.maxsize)
        import matplotlib.pyplot as plt
        plt.scatter(ground_truth[20, 20, :], ground_truth[20, 23, :], alpha=0.3)
        plt.show()

    def pred(self, x):
        """Given a (40x1) vector, return a predicted 40x40 array"""
        # x = gaussian_filter(x, radius=2, sigma=1)
        pred = self.model.predict(x).reshape((40, 39))
        return self.insert_diagonal(pred, x.flatten())

    def get_params(self, t):
        """Returns array of params of the models for the (t,t)'th coordinate"""
        params = np.zeros(shape=(40, 40))
        for i in range(40):
            for j in range(40):
                coefs = self.models[i][j].coef_
                params[i, j] = coefs[t]
        return params

    def save_model(self, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_mode(self, file_name):
        with open(file_name, 'rb') as handle:
            self.model = pickle.load(handle)

    @staticmethod
    def get_diagonal_indices(n, m):
        arr = np.arange(n * m).reshape(n, m)
        # Initialize an empty list to store coordinates
        coordinates = []
        for i in range(min(n, m)):
            coordinates.append(arr[i, i])
        for d in range(1, max(n, m)):
            for i in range(min(n, m)):
                if i + d < n:
                    coordinates.append(arr[i + d, i])
            for i in range(min(n, m)):
                if i + d < m:
                    coordinates.append(arr[i, i + d])
        return coordinates

    # @staticmethod
    # def get_diagonal_indices_list():
    #     diagonal_indices = []
    #     for shift in range(-40 + 1, 40):
    #         diagonal_indices.append(
    #             [i * 40 + (i + shift) for i in range(max(0, -shift), min(40, 40 - shift))])
    #
    #     diagonal_indices.sort(key=len)
    #     diagonal_indices.reverse()
    #     flat_diagonal_indices = []
    #     for row in diagonal_indices:
    #         flat_diagonal_indices += row
    #     return flat_diagonal_indices
    #
    # @staticmethod
    # def get_diagonal_indices_list_no_diag():
    #     diagonal_indices = []
    #     for shift in range(-40 + 1, 40):
    #         diagonal_indices.append(
    #             [i * 39 + (i + shift) if i + shift < 40 else i * 39 + (i + shift - 1) for i in
    #              range(max(0, -shift), min(40, 40 - shift))])
    #
    #     diagonal_indices.sort(key=len)
    #     diagonal_indices.reverse()
    #     flat_diagonal_indices = []
    #     for row in diagonal_indices:
    #         flat_diagonal_indices += row
    #     return flat_diagonal_indices

    @staticmethod
    def remove_diagonal(arr):
        n = arr.shape[0]
        mask = np.ones(arr.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        return arr[mask].reshape(n, n - 1)

    @staticmethod
    def insert_diagonal(arr, diag):
        n = arr.shape[0]
        result = np.zeros((n, n))
        mask = np.ones(shape=(n, n), dtype=bool)
        np.fill_diagonal(mask, 0)
        np.fill_diagonal(result, diag)
        result[mask] = arr.flatten()
        return result
