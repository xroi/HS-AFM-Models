from datasets.series.line_series_dataset import LineSeriesDataset
from datasets.series.line_series_torch_wrapper import LineSeriesTorchWrapper
import numpy as np
import pickle
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader


class COMNeuralNet(nn.Module):
    def __init__(self, device, num_layers, hidden_size):
        super(COMNeuralNet, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.h = None
        self.refresh_h0()
        self.rnn = nn.RNN(input_size=40, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          dtype=torch.float64).to(device)
        self.fc1 = self.fc1 = nn.Linear(self.hidden_size, 2, dtype=torch.float64).to(device)

    def forward(self, x):
        x = x[None, ...]
        out, self.h = self.rnn(x, self.h)
        out = self.fc1(out)
        return out

    def refresh_h0(self):
        self.h = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float64).to(self.device)


class COMNeuralNetTrainer():
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    DEVICE = "cuda"
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_HIDDEN_RNN_lAYERS = 2
    HIDDEN_SIZE = 80

    # dataset parameters
    COM_RADIUS_AROUND_CENTER = 11
    SPATIAL_WINDOW_SIZE = 7
    TEMPORAL_WINDOW_SIZE = 20
    MAX_SEQ_lEN = 50
    CENTER_COM_COEF = -19.5

    def __init__(self):
        self.model = COMNeuralNet(self.DEVICE, self.NUM_HIDDEN_RNN_lAYERS, self.HIDDEN_SIZE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.loss_func = nn.MSELoss()

    def train(self):

        # Set up the dataset
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40", 100, line_y=20, get_back_frames=False,
                                    temporal_window_size=self.TEMPORAL_WINDOW_SIZE,
                                    spatial_window_size=self.SPATIAL_WINDOW_SIZE)
        torch_dataset = LineSeriesTorchWrapper(dataset, self.COM_RADIUS_AROUND_CENTER, max_seq_len=self.MAX_SEQ_lEN)
        # train_dataloader = DataLoader(dataset=torch_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        # Train the model
        for epoch in range(self.NUM_EPOCHS):
            loss = 0
            for i in range(len(torch_dataset)):
                x, y = torch_dataset[i]
                # Moved model to device, so tensors have to be moved too.
                x = x.to(self.DEVICE)
                y = y.to(self.DEVICE) + self.CENTER_COM_COEF
                loss += self.train_single_sequence(x, y)
            print(f"epoch {epoch} mean loss = {loss / len(torch_dataset)}")
        return

    def test(self):
        # Set up the dataset
        dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40/test", 100, line_y=20, get_back_frames=False,
                                    temporal_window_size=self.TEMPORAL_WINDOW_SIZE,
                                    spatial_window_size=self.SPATIAL_WINDOW_SIZE)
        torch_dataset = LineSeriesTorchWrapper(dataset, self.COM_RADIUS_AROUND_CENTER, max_seq_len=self.MAX_SEQ_lEN)
        # train_dataloader = DataLoader(dataset=torch_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        # Train the model
        loss = 0
        for i in range(len(torch_dataset)):
            x, y = torch_dataset[i]
            # Moved model to device, so tensors have to be moved too.
            x = x.to(self.DEVICE)
            y = y.to(self.DEVICE) + self.CENTER_COM_COEF
            pred_y = self.predict(x)
            loss += self.loss_func(y, pred_y)
        return

    def train_single_sequence(self, line_scans_seq, com_ground_truths):
        self.model.refresh_h0()
        # output = torch.zeros(1, 2).to(self.DEVICE)
        seq_loss = 0
        for i in range(line_scans_seq.size()[0]):
            self.optimizer.zero_grad()
            o = self.model.forward(line_scans_seq[i, :])
            # output = torch.cat((output, o), 0)
            loss = self.loss_func(o, com_ground_truths[i, :])
            loss.backward()
            self.optimizer.step()
            seq_loss += loss
            seq_loss = seq_loss.detach()
        output = output[1:, :]
        mean_seq_loss = seq_loss.item() / line_scans_seq.size()[0]
        print(mean_seq_loss)
        return mean_seq_loss

    def predict(self, line_scans_seq):
        self.model.refresh_h0()
        output = torch.zeros(1, 2).to(self.DEVICE)
        for i in range(line_scans_seq.size()[0]):
            o = self.model.forward(line_scans_seq[i, :])
            output = torch.cat((output, o), 0)
        output = output[1:, :]
        return output

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()
