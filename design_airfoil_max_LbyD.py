import torch
from torch import nn

from neural_net.utils import set_learning_rate, print_net_performance
from neural_net.net_def import NeuralNetwork


## Initialize random input that is to be optimized
X = torch.rand(1, 3, requires_grad = True)


## Load the pre-trained network
# Initialize network architecture
input_dim, hidden_dim, layer_count = 3, 10, 3
xfoil_net = NeuralNetwork(input_dim, hidden_dim, layer_count)
# Load latest checkpoint and set state dict of model
checkpoint = torch.load('neural_net/checkpoints/latest.pth')
xfoil_net.load_state_dict(checkpoint['model'])


## Define the loss function
def loss_fn(output):
    loss = -1 * torch.mean(output)
    return loss

## Define an optimizer
optimizer = torch.optim.Adam([X])


## Perform input optimization
# Set the training properties
epochs = 1000
print_cost_every = 100
learning_rate = .01

# Set learning rate
set_learning_rate(optimizer, learning_rate)


# Run the training loop
for epoch in range(1, epochs + 1):
    # Run the forward pass and calculate the prediction
    Y_pred = xfoil_net(X)

    # Compute the loss
    loss = loss_fn(Y_pred)

    # Run the backward pass and calculate the gradients
    loss.backward()

    # Take an update step and then zero out the gradients
    optimizer.step()
    optimizer.zero_grad()

    # Print training progress
    if epoch % print_cost_every == 0 or epoch == 1:
        J_train = loss.item()

        # Print the current performance
        # print_net_performance(epochs = epochs, epoch = epoch, J_train = J_train)
        num_digits = len(str(epochs))
        print(f'Epoch: [{epoch:{num_digits}}/{epochs}]. Train Cost: {J_train:11.6f}. Y_pred: {Y_pred.item():.2f}')