import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.nn as nn
from temp_trajectory_dataset import *
from trajectory_neural_net import *


def main():
    batch_size = 50
    learning_rate = 0.001
    num_epochs = 5

    # Set up the datasets
    train_dataset = TrajectoryDataset("train_pickles/", 7)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    n_total_steps = len(train_dataloader)

    # Choose device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = TrajectoryNeuralNet().to(device)

    # Loss and optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_dataloader):
            # Moved model to device, so tensors have to be moved too.
            x = x.to(device)
            y = y.to(device)

            # Forward pass and loss calculation
            y_hat = model(x)
            loss = loss_func(y_hat, y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print info
            if (i + 1) % 200 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    pred = np.zeros(shape=(36, 36, 2))
    dist = np.zeros(shape=(36, 36))
    actual = np.zeros(shape=(36, 36, 2))
    # Test the model
    # with torch.no_grad():
    test_dataset = TrajectoryDataset("test_pickles/", 1)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    x_i = 0
    y_i = 0
    for i, (x, y) in enumerate(test_dataloader):
        if i % 36 == 0:
            y_i += 1
            x_i = 0
        if y_i == 36:
            break
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)

        actual[x_i, y_i, :] = y.detach().numpy()
        pred[x_i, y_i, :] = y_hat.detach().numpy()
        dist[x_i, y_i] = loss_func(y_hat, y).detach().numpy()
        x_i += 1

    with open("test_pickles/1.pickle", 'rb') as f:
        pickle1 = pickle.load(f)
    raster = pickle1["rasterized_maps"][0][2:38, 2:38]
    pass


if __name__ == "__main__":
    main()
