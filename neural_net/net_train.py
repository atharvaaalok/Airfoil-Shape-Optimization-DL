import os.path

import torch
from torch import nn
import numpy as np

from net_def import NeuralNetwork
from utils import set_learning_rate, print_net_performance


## Get the data
train_data = np.load('../data/generated_airfoils/train/airfoil_data.npz')
X_all = train_data['P']
Y_all = train_data['L_by_D'].reshape(-1, 1)
X_train = torch.from_numpy(X_all).to(torch.float32)
Y_train = torch.from_numpy(Y_all).to(torch.float32)
val_data = np.load('../data/generated_airfoils/dev/airfoil_data.npz')
X_all = val_data['P']
Y_all = val_data['L_by_D'].reshape(-1, 1)
X_val = torch.from_numpy(X_all).to(torch.float32)
Y_val = torch.from_numpy(Y_all).to(torch.float32)

m_train = X_train.shape[0]


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
epochs = 100000
print_cost_every = 10000
B = 64
learning_rate = 0.001

# Set learning rate
set_learning_rate(optimizer, learning_rate)

# Load saved model if available
if os.path.exists('checkpoints/latest.pth'):
    checkpoint = torch.load('checkpoints/latest.pth')
    xfoil_net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    total_epochs = checkpoint['total_epochs']
else:
    total_epochs = 0


# Run the training loop
for epoch in range(total_epochs + 1, total_epochs + epochs + 1):
    # Set network to training mode
    xfoil_net.train()

    # Select a mini-batch of data
    idx = torch.randint(0, m_train, (B,))
    x = X_train[idx, :]
    y = Y_train[idx, :]

    # Run the forward pass and calculate the predictions
    Y_pred = xfoil_net(x)

    # Compute the loss
    loss = MSELoss_fn(Y_pred, y)

    # Run the backward pass and calculate the gradients
    loss.backward()

    # Take an update step and then zero out the gradients
    optimizer.step()
    optimizer.zero_grad()

    # Print training progress
    if epoch % print_cost_every == 0 or epoch == total_epochs + 1:
        J_train = loss.item()

        #  Evaluate current model on validation data
        xfoil_net.eval()
        Y_pred = xfoil_net(X_val)
        loss = MSELoss_fn(Y_pred, Y_val)
        J_val = loss.item()

        # Print the current performance
        print_net_performance(epochs = total_epochs + epochs, epoch = epoch, J_train = J_train, J_val = J_val)

        # Create checkpoint and save the model
        checkpoint = {
            'total_epochs': epoch,
            'model': xfoil_net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        # Save the model twice: once on its own and once in the latest model file
        torch.save(checkpoint, f'checkpoints/xfoil_net_{epoch}_Jtrain_{J_train:.2e}_Jval_{J_val:.2e}.pth')
        torch.save(checkpoint, f'checkpoints/latest.pth')