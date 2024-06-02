from datasets.series.line_series_dataset import LineSeriesDataset
from datasets.series.line_series_torch_wrapper import LineSeriesTorchWrapper
import numpy as np
import pickle
import torch.nn as nn
import torch.optim
import optuna
from torch.utils.data import DataLoader


class COMNeuralNet(nn.Module):
    def __init__(self, device, num_layers, hidden_size, bidirectional=False):
        super(COMNeuralNet, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.D = 2 if self.bidirectional else 1

        self.rnn = nn.RNN(input_size=40, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          dtype=torch.float64, batch_first=True, bidirectional=self.bidirectional).to(device)
        self.fc1 = self.fc1 = nn.Linear(self.hidden_size * self.D, 3, dtype=torch.float64).to(device)

    def forward(self, x):
        h_0 = torch.randn(self.num_layers * self.D, 1, self.hidden_size, dtype=torch.float64).to(self.device)
        x = x[None, ...]
        out, h_n = self.rnn(x, h_0)
        out = self.fc1(out)
        return out


class COMNeuralNetTrainer:
    BATCH_SIZE = 1
    NUM_EPOCHS = 5
    DEVICE = "cuda"
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset parameters
    COM_RADIUS_AROUND_CENTER = 11
    SPATIAL_WINDOW_SIZE = 5
    TEMPORAL_WINDOW_SIZE = 10
    # MAX_SEQ_lEN = 38  # 38 or 19
    CENTER_COM_COEF = -19.5
    CENTER_SCAN_COEF = -47

    def __init__(self, learning_rate, num_layers, hidden_size, bidirectional, max_seq_len):
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.max_seq_len = max_seq_len

        self.model = COMNeuralNet(self.DEVICE, self.num_layers, self.hidden_size, bidirectional=self.bidirectional)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def train(self):

        # Set up the dataset
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40", 100, line_y=20, get_back_frames=False,
                                    temporal_window_size=self.TEMPORAL_WINDOW_SIZE,
                                    spatial_window_size=self.SPATIAL_WINDOW_SIZE)
        torch_dataset = LineSeriesTorchWrapper(dataset, self.COM_RADIUS_AROUND_CENTER, max_seq_len=self.max_seq_len)
        # train_dataloader = DataLoader(dataset=torch_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        # Train the model
        for epoch in range(self.NUM_EPOCHS):
            loss = 0
            for i in range(len(torch_dataset)):
                x, y = torch_dataset[i]
                x = (x + self.CENTER_SCAN_COEF)
                y[:, 0:2] = (y[:, 0:2] + self.CENTER_COM_COEF)
                y[:, 2] = (y[:, 2] + self.CENTER_SCAN_COEF)
                # Moved model to device, so tensors have to be moved too.
                x = x.to(self.DEVICE)
                y = y.to(self.DEVICE)
                loss += self.train_single_sequence(x, y)
            print(f"epoch {epoch} mean loss = {loss / len(torch_dataset)}")
            # self.model.eval()
            # self.test()
            # self.model.train()
        return

    def test(self):
        # Set up the dataset
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40/validation", 100, line_y=20,
                                    get_back_frames=False,
                                    temporal_window_size=self.TEMPORAL_WINDOW_SIZE,
                                    spatial_window_size=self.SPATIAL_WINDOW_SIZE)
        torch_dataset = LineSeriesTorchWrapper(dataset, self.COM_RADIUS_AROUND_CENTER, max_seq_len=38)
        # train_dataloader = DataLoader(dataset=torch_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        a0_gts, a0_preds, a1_gts, a1_preds = np.array([0]), np.array([0]), np.array([0]), np.array([0])
        x_loss, y_loss = 0, 0
        for i in range(len(torch_dataset)):
            x, y = torch_dataset[i]
            # Moved model to device, so tensors have to be moved too.
            x = (x + self.CENTER_SCAN_COEF)
            y[:, 0:2] = (y[:, 0:2] + self.CENTER_COM_COEF)
            y[:, 2] = (y[:, 2] + self.CENTER_SCAN_COEF)
            x = x.to(self.DEVICE)
            y = y.to(self.DEVICE)
            pred_y = self.predict(x)
            x_loss += self.loss_func(y[:, 0], pred_y[0, :, 0])
            y_loss += self.loss_func(y[:, 1], pred_y[0, :, 1])
            a0_gts = np.concatenate((a0_gts, y[:, 0].cpu().detach().numpy()))
            a1_gts = np.concatenate((a1_gts, y[:, 1].cpu().detach().numpy()))
            a0_preds = np.concatenate((a0_preds, pred_y[0, :, 0].cpu().detach().numpy()))
            a1_preds = np.concatenate((a1_preds, pred_y[0, :, 1].cpu().detach().numpy()))
        print([x_loss / len(torch_dataset), y_loss / len(torch_dataset)])
        import matplotlib.pyplot as plt
        m, b = np.polyfit(a1_gts, a1_preds, 1)
        plt.scatter(a1_gts, a1_preds, alpha=0.2)
        plt.plot(a1_gts, m * a1_gts + b, color="orange")
        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        ax = plt.gca()
        plt.axis('equal')
        plt.show()
        return x_loss / len(torch_dataset) + y_loss / len(torch_dataset)

    def train_single_sequence(self, line_scans_seq, com_ground_truths):
        # self.model.refresh_h0()
        self.optimizer.zero_grad()
        o = self.model.forward(line_scans_seq)
        loss = self.loss_func(o[0, :, :], com_ground_truths)
        loss.backward()
        self.optimizer.step()
        # print(loss.item())
        return loss

    def predict(self, line_scans_seq):
        # self.model.refresh_h0()
        return self.model.forward(line_scans_seq)

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()

    def test_visually(self):
        INDEX = 0
        from visualizations.output_video import output_video
        from models.statistical.movment_tracker import MovementSeries
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40/test", 100, line_y=20, get_back_frames=False,
                                    temporal_window_size=self.TEMPORAL_WINDOW_SIZE,
                                    spatial_window_size=self.SPATIAL_WINDOW_SIZE)
        torch_dataset = LineSeriesTorchWrapper(dataset, self.COM_RADIUS_AROUND_CENTER, max_seq_len=38)
        x, y = torch_dataset[INDEX]
        x = (x + self.CENTER_SCAN_COEF)
        y[:, 0:2] = (y[:, 0:2] + self.CENTER_COM_COEF)
        y[:, 2] = (y[:, 2] + self.CENTER_SCAN_COEF)
        x = x.to(self.DEVICE)
        y = y.to(self.DEVICE)
        pred_y = self.predict(x)
        movement_series_list = [MovementSeries(0), MovementSeries(0)]
        for j in range(y.shape[0]):
            movement_series_list[0].add_position(y[j, 0] - self.CENTER_COM_COEF, y[j, 1] - self.CENTER_COM_COEF)
            movement_series_list[1].add_position(pred_y[0, j, 0] + 20,
                                                 pred_y[0, j, 1] + 20)
        non_rasters = torch_dataset.get_matching_non_rasters(INDEX)
        output_video(non_rasters, f"test_new2.mp4", 40, 55, 1000, 1000, "jet",
                     timestamp_step=4, add_legend=True, crop_from_sides_px=0, draw_inner_circle_r=11,
                     draw_outer_circle_r=18.5, max_frames=3200, movement_series_list=movement_series_list,
                     frames_per_second=1, add_tip_position=False)


class COMNetHyperparameterOptimizer:
    def optimize_hyperparameters(self):
        def objective(trial: optuna.trial):
            hidden_size = trial.suggest_int('hidden_size', 10, 200)
            num_layers = trial.suggest_int('num_layers', 1, 5)
            max_seq_len = trial.suggest_categorical('max_seq_len', [19, 38])
            bidirectional = trial.suggest_categorical('bidirectional', [True, False])
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)

            com_net = COMNeuralNetTrainer(hidden_size=hidden_size,
                                          num_layers=num_layers,
                                          max_seq_len=max_seq_len,
                                          bidirectional=bidirectional,
                                          learning_rate=learning_rate)
            com_net.train()
            loss = com_net.test()
            return loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        print(study.best_params)

        # {'hidden_size': 120, 'num_layers': 1, 'max_seq_len': 19, 'bidirectional': True, 'learning_rate':
        # 0.005695743523148093}
        # 5.977789333984168
