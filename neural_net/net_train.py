import torch
from torch import nn
import numpy as np

from net_def import NeuralNetwork


## Get the data
X_all = np.load('../data/generated_airfoils/airfoils_original.npy')
Y_all = np.load('../data/generated_airfoils/L_by_D_original.npy').reshape(-1, 1)
X_train = torch.from_numpy(X_all).to(torch.float32)
Y_train = torch.from_numpy(Y_all).to(torch.float32)


## Initialize the network
# Set network properties
input_dim, hidden_dim = 402, 5
xfoil_net = NeuralNetwork(input_dim, hidden_dim)


## Define the loss function
MSELoss_fn = nn.MSELoss()

## Define an optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(xfoil_net.parameters(), lr = learning_rate)


## Train the network
# Set the training properties
epochs = 1000
print_cost_every = 100
B = 64


for epoch in range(1, epochs + 1):
    # Run the forward pass and calculate the predictions
    Y_pred = xfoil_net(X_train)

    # Compute the loss
    loss = MSELoss_fn(Y_pred, Y_train)

    # Run the backward pass and calculate the gradients
    loss.backward()

    # Take an update step and then zero out the gradients
    optimizer.step()
    optimizer.zero_grad()

    # Print training progress
    if epoch % print_cost_every == 0 or epoch == 1:
        J_train = loss.item()
        print(J_train)