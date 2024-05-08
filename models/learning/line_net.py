import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.series.line_series_dataset import LineSeriesDataset
from datasets.series.line_series_torch_wrapper import LineSeriesTorchWrapper


class LineNeuralNet(nn.Module):
    def __init__(self):
        super(LineNeuralNet, self).__init__()
        self.fc1 = nn.Linear(40 * 40, 40 * 40, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(40 * 40, 40 * 40, dtype=torch.float64)
        self.fc3 = nn.Linear(40 * 40, 40 * 40, dtype=torch.float64)

    def forward(self, x):
        x = x.view(-1, 40 * 40)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = out.view(-1, 40, 40)

        return out

    @staticmethod
    def train_network(regression_model):
        batch_size = 32
        learning_rate = 0.001
        num_epochs = 2

        # Set up the datasets
        orig_dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40", 75, line_y=20, get_back_frames=False)
        torch_dataset = LineSeriesTorchWrapper(orig_dataset, regression_model)
        train_dataloader = DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)
        n_total_steps = len(train_dataloader)

        # Choose device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        model = LineNeuralNet().to(device)

        # Loss and optimizer
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(train_dataloader):
                # Moved model to device, so tensors have to be moved too.
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                y_hat = model(x)

                # Calculate Loss
                loss = loss_func(y_hat - y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Print info
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
        return model
