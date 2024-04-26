from sklearn.linear_model import LinearRegression
from datasets.series.line_series_dataset import LineSeriesDataset
import numpy as np
import pickle


class LineRegression():
    def __init__(self):
        self.models = [[0] * 40 for _ in range(40)]

    def train_regression_models(self):
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40", 50, line_y=20, get_back_frames=False)
        stacked_data = dataset.get_stacked_data()
        regression_X = dataset.get_regression_formatted_data_(stacked_data)
        for i in range(40):
            for j in range(40):
                model = LinearRegression()
                model.fit(regression_X, stacked_data[i, j, :])
                self.models[i][j] = model

    def pred(self, x):
        """Given a rasterized (40x1) vector, return a predicted 40x40 array"""
        pred = np.zeros(shape=(40, 40))
        for i in range(40):
            for j in range(40):
                pred[i, j] = self.models[i][j].predict(x)
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
