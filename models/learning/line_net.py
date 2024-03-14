import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.series.line_series_dataset import LineSeriesDataset
from datasets.series.line_series_torch_wrapper import LineSeriesTorchWrapper


class LineNeuralNet(nn.Module):
    def __init__(self):
        super(LineNeuralNet, self).__init__()
        self.l1 = nn.Linear(1600, 1600, dtype=torch.float64)
        self.r1 = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=8, stride=1, padding="same", dtype=torch.float64)
        self.r2 = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 8), stride=1, padding="same",
                            dtype=torch.float64)
        self.r3 = nn.ReLU()
        self.l2 = nn.Linear(1600, 1600, dtype=torch.float64)

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.l1(out)
        out = torch.unflatten(out, 1, (40, 40))
        out = self.r1(out)
        out = out[:, None, :, :]  # required for conv2d
        out = self.c1(out)
        out = self.r2(out)
        out = self.c2(out)
        out = self.r3(out)
        out = torch.flatten(out, start_dim=1)
        out = self.l2(out)
        out = torch.unflatten(out, 1, (40, 40))
        # remove dimension of size 1
        out = torch.squeeze(out)
        # min-max normalize the result
        # out_min, out_max = out.min(), out.max()
        # new_min, new_max = 0, 1
        # out = (out - out_min) / (out_max - out_min) * (new_max - new_min) + new_min
        out = torch.softmax()
        return out

    @staticmethod
    def train_network():
        batch_size = 64
        learning_rate = 0.01
        num_epochs = 5

        # Set up the datasets
        orig_dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40", 60, line_y=20)
        torch_dataset = LineSeriesTorchWrapper(orig_dataset)
        train_dataloader = DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)
        n_total_steps = len(train_dataloader)

        # Choose device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        model = LineNeuralNet().to(device)

        # Loss and optimizer
        loss_func = nn.L1Loss()
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
                loss = loss_func(y_hat, y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Print info
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
        return model
