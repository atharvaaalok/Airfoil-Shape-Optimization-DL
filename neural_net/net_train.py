import torch
from torch import nn
import numpy as np

from net_def import NeuralNetwork
from utils import set_learning_rate, print_net_performance


## Get the data
X_all = np.load('../data/generated_airfoils/airfoils_original.npy')
Y_all = np.load('../data/generated_airfoils/L_by_D_original.npy').reshape(-1, 1)
X_train = torch.from_numpy(X_all).to(torch.float32)
Y_train = torch.from_numpy(Y_all).to(torch.float32)
X_val = X_train
Y_val = Y_train


## Initialize the network
# Set network properties
input_dim, hidden_dim = 402, 5
xfoil_net = NeuralNetwork(input_dim, hidden_dim)


## Define the loss function
MSELoss_fn = nn.MSELoss()

## Define an optimizer
optimizer = torch.optim.Adam(xfoil_net.parameters())


## Train the network
# Set the training properties
epochs = 1000
print_cost_every = 100
B = 64
learning_rate = 1e-6

# Set learning rate
set_learning_rate(optimizer, learning_rate)


# Run the training loop
for epoch in range(1, epochs + 1):
    # Set network to training mode
    xfoil_net.train()

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

        #  Evaluate current model on validation data
        xfoil_net.eval()
        Y_pred = xfoil_net(X_val)
        loss = MSELoss_fn(Y_pred, Y_val)
        J_val = loss.item()

        # Print the current performance
        print_net_performance(epochs = epochs, epoch = epoch, J_train = J_train, J_val = J_val)

        # Create checkpoint and save the model
        checkpoint = {
            'model': xfoil_net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoints/xfoil_net_{epoch}.pth')