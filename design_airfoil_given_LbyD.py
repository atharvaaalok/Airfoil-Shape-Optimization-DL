import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from data.airfoil_utils.generate_airfoil_parameterization import fit_catmullrom, get_catmullrom_points
from data.airfoil_utils.compute_L_by_D import compute_L_by_D
from neural_net.net_def import NeuralNetwork
# Import pre-trained network
from neural_net.trained_nets.FullData_300nodes_10layers.net_load import xfoil_net


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

## Initialize random input that is to be optimized
airfoil_name = 'naca0006'
X = np.loadtxt(f'data/airfoil_database/airfoils/{airfoil_name}.dat')
X_centroid = np.mean(X, axis = 0)
X = X - X_centroid
X = fit_catmullrom(X.flatten(), num_control_pts = 12)
X = X.to(torch.float32).to(device)

X = X.reshape(-1, 2)
X_list = []
for i in range(X.shape[0]):
    X_list.append(X[i])

# Extract the portion of X to be optimized, leave the first and the last point
X_opt = torch.vstack(X_list[1:-1])
X_opt.requires_grad = True

X = torch.vstack([X_list[0], X_opt, X_list[-1]]).reshape(1, -1)


# Fix the L by D ratio for which we design the airfoil
Y = torch.tensor([75.0]).reshape(1, -1).to(device)


## Load the pre-trained network
# Initialize network architecture
xfoil_net = xfoil_net.to(device)


## Define the loss function
MSELoss_fn = nn.MSELoss()

## Define an optimizer
learning_rate = 0.0001
weight_decay = 0
optimizer = torch.optim.Adam([X_opt], lr = learning_rate, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)


## Perform input optimization
# Set the training properties
epochs = 500
print_cost_every = 1

# Set model to evaluation model to allow batch normalization to work with single input
xfoil_net.eval()


# Run the training loop
for epoch in range(1, epochs + 1):
    # Reconstruct input X after every gradient step
    X = torch.vstack([X_list[0], X_opt, X_list[-1]]).reshape(1, -1)

    # Run the forward pass and calculate the prediction
    with torch.set_grad_enabled(True):
        Y_pred = xfoil_net(X)

        # Compute the loss
        loss = MSELoss_fn(Y_pred, Y)

    # Run the backward pass and calculate the gradients
    loss.backward()

    # Take an update step and then zero out the gradients
    optimizer.step()
    optimizer.zero_grad()

    # Print training progress
    if epoch % print_cost_every == 0 or epoch == 1:
        J_train = loss.item()

        # Compute L by D predicted by Xfoil
        X_fit = get_catmullrom_points(X.detach().reshape(-1, 2), num_sample_pts = 201).detach().numpy()
        Y_xfoil = compute_L_by_D(X_fit.flatten())

        # Print the current performance
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch:{num_digits}}/{epochs}]. Train Cost: {J_train:11.6f}. Y: {Y.item():.2f}. Y_pred: {Y_pred.item():.2f}. Y_xfoil: {Y_xfoil:.2f}')

    
    scheduler.step(J_train)

# Get coordinates on the boundary
X_fit = get_catmullrom_points(X.detach().reshape(-1, 2), num_sample_pts = 201)
# Plot the airfoil
plt.plot(X_fit[:, 0], X_fit[:, 1])
plt.axis('equal')
plt.savefig('Final_airfoil.png', dpi = 600)