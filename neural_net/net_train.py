import os.path

import torch
from torch import nn
from torch.utils.data import DataLoader

from net_def import NeuralNetwork
from net_data import AirfoilDataset
from utils import set_learning_rate, print_net_performance
from utils import red, cyan, color_end


def train_loop(dataloader, model, loss_fn, optimizer, verbose = False):
    num_batches = len(dataloader)
    num_digits = len(str(num_batches))

    # Set the model to training mode
    model.train()

    # Run the training loop
    for batch, (X, Y) in enumerate(dataloader):
        # Run the forward pass
        Y_pred = model(X)

        # Compute the loss
        loss = loss_fn(Y_pred, Y)

        # Run the backward pass and calculate the gradients
        loss.backward()

        # Take an update step and then zero out the gradients
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            if batch % (num_batches // 5) == 0:
                loss = loss.item()
                print(f'{cyan}Train Loss:{color_end} [{batch + 1:{num_digits}}/{num_batches}] {loss:20.6f}')
    if verbose:
        print()


def dev_loop(dataloader, model, loss_fn, verbose = False):
    num_batches = len(dataloader)
    test_loss = 0

    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model with torch.no_grad() to ensure no gradients are computed
    with torch.no_grad():
        for X, Y in dataloader:
            Y_pred = model(X)
            test_loss += loss_fn(Y_pred, Y).item()
    
    test_loss = test_loss / num_batches
    if verbose:
        print(f'{cyan}Valid Loss:{color_end} {test_loss:20.6f}')
        print()


## Get the data
train_filepath = '../data/generated_airfoils/train/airfoil_data.npz'
dev_filepath = '../data/generated_airfoils/dev/airfoil_data.npz'

# Create data set instances for training and dev data
train_dataset = AirfoilDataset(train_filepath)
dev_dataset = AirfoilDataset(dev_filepath)

# Fix the batch size
B = 64

# Create data loader instances for training and dev sets
train_dataloader = DataLoader(train_dataset, batch_size = B, shuffle = True)
dev_dataloader = DataLoader(dev_dataset, batch_size = B, shuffle = True)


## Initialize the network
# Set network properties
input_dim, hidden_dim, layer_count = 24, 10, 3
xfoil_net = NeuralNetwork(input_dim, hidden_dim, layer_count)


## Define the loss function
MSELoss_fn = nn.MSELoss()


## Define an optimizer
optimizer = torch.optim.Adam(xfoil_net.parameters())


## Train the network
# Set the training properties
epochs = 1000
learning_rate = 0.001

# Set learning rate
set_learning_rate(optimizer, learning_rate)


# Train the network
for epoch in range(1, epochs):
    
    if epoch % (epochs // 10) == 0 or epoch == 1:
        verbose = True
    else:
        verbose = False

    if verbose:
        print(f'{red}Epoch {epoch}{color_end}\n' + 40 * '-')

    # Run the training loop
    train_loop(train_dataloader, xfoil_net, MSELoss_fn, optimizer, verbose)

    # Run the validation loop
    dev_loop(dev_dataloader, xfoil_net, MSELoss_fn, verbose)


print('Finished Training!')