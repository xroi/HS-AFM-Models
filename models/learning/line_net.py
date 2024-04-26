import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.series.line_series_dataset import LineSeriesDataset
from datasets.series.line_series_torch_wrapper import LineSeriesTorchWrapper


class LineNeuralNet(nn.Module):
    def __init__(self):
        super(LineNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding="same", dtype=torch.float64)
        self.bn1 = nn.BatchNorm2d(10, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding="same", dtype=torch.float64)
        self.bn2 = nn.BatchNorm2d(20, dtype=torch.float64)
        self.relu2 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.15)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5, stride=1, padding="same", dtype=torch.float64)
        self.bn3 = nn.BatchNorm2d(40, dtype=torch.float64)
        self.relu3 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.15)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding="same", dtype=torch.float64)
        self.rnn = nn.RNN(40, 40, 2, batch_first=True, dtype=torch.float64)
        self.fc = nn.Linear(40 * 40, 40 * 40, dtype=torch.float64)

    def forward(self, x):
        x = x.unsqueeze(1)  # add a channel dimension
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.do1(self.relu2(self.bn2(self.conv2(x))))
        x = self.do2(self.relu3(self.bn3(self.conv3(x))))
        x = x.permute((0, 3, 2, 1))
        x = self.conv4(x)
        x = x.squeeze(1)

        x = torch.transpose(x, 1, 2)
        h0 = torch.zeros(2, x.shape[0], 40, dtype=torch.float64)
        x, _ = self.rnn(x, h0)

        x = torch.flatten(x, 1, 2)
        x = self.fc(x)
        x = torch.unflatten(x, 1, (40, 40))

        return x

    @staticmethod
    def train_network():
        batch_size = 64
        learning_rate = 0.01
        num_epochs = 10

        # Set up the datasets
        orig_dataset = LineSeriesDataset("temp_datasets/mini_200uM_100ns_40x40", 75, line_y=20)
        torch_dataset = LineSeriesTorchWrapper(orig_dataset)
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
                loss = loss_func(y_hat, y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Print info
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
        return model
