import sys

from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import PolynomialFeatures
from datasets.series.line_series_dataset import LineSeriesDataset
import numpy as np
import pickle


class LineRegression1:
    def __init__(self):
        self.models = [[0] * 40 for _ in range(40)]

    def train_regression_models(self):
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40", 75, line_y=20, get_back_frames=False)
        stacked_data = dataset.get_stacked_data()
        regression_X = dataset.get_regression_formatted_data_(stacked_data)  # n_samples x n_features
        # poly = PolynomialFeatures(2, interaction_only=True)
        # regression_X = poly.fit_transform(regression_X)
        for i in range(40):
            for j in range(40):
                print(i, " ", j)
                model = Ridge(alpha=4096)
                model.fit(regression_X, stacked_data[i, j, :])
                self.models[i][j] = model

    def test_model(self):
        # pickles 75 to 80
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
        plt.scatter(pred_values[20, 23, :], residuals[20, 23, :])
        plt.show()

    def pred(self, x):
        """Given a rasterized (40x1) vector, return a predicted 40x40 array"""
        pred = np.zeros(shape=(40, 40))
        # poly = PolynomialFeatures(2, interaction_only=True)
        # x = poly.fit_transform(x)
        for i in range(40):
            for j in range(40):
                pred[i, j] = self.models[i][j].predict(x)
        # pred = gaussian_filter(pred, radius=1, sigma=0.5)
        return pred

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
            pickle.dump(self.models, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_mode(self, file_name):
        with open(file_name, 'rb') as handle:
            self.models = pickle.load(handle)
