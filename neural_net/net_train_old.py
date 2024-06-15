import os.path

import torch
from torch import nn
from torch.utils.data import DataLoader

from net_def import NeuralNetwork
from net_data import AirfoilDataset
from utils import set_learning_rate, train_loop_old, dev_loop_old, red, color_end


## Get the data
train_filepath = '../data/generated_airfoils/train/airfoil_data.npz'
dev_filepath = '../data/generated_airfoils/dev/airfoil_data.npz'

# Create data set instances for training and dev data
train_dataset = AirfoilDataset(train_filepath)
dev_dataset = AirfoilDataset(train_filepath)

# Fix the batch size
B = 900

# Create data loader instances for training and dev sets
train_dataloader = DataLoader(train_dataset, batch_size = B, shuffle = True)
dev_dataloader = DataLoader(dev_dataset, batch_size = B, shuffle = True)


## Initialize the network
# Set network properties
input_dim, hidden_dim, layer_count = 24, 30, 4
xfoil_net = NeuralNetwork(input_dim, hidden_dim, layer_count)


## Define the loss function
MSELoss_fn = nn.MSELoss()


## Define an optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(xfoil_net.parameters(), lr = learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.9)


## Train the network
# Set the training properties
epochs = 100000
print_cost_every = 10000


# # Load saved model if available
# if os.path.exists('checkpoints/latest.pth'):
#     checkpoint = torch.load('checkpoints/latest.pth')
#     xfoil_net.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     total_epochs = checkpoint['total_epochs']
# else:
#     total_epochs = 0
total_epochs = 0

# Train the network
for epoch in range(total_epochs + 1, total_epochs + epochs + 1):

    verbose = True if epoch % print_cost_every == 0 or epoch == total_epochs + 1 else False
    save = verbose

    if verbose:
        print(f'{red}Epoch {epoch}{color_end}\n' + 40 * '-')
        # print(scheduler.get_last_lr())


    # Run the training loop
    train_loop_old(train_dataloader, xfoil_net, MSELoss_fn, optimizer, verbose = False)


    # Run the validation loop
    J_val = dev_loop_old(dev_dataloader, xfoil_net, MSELoss_fn, verbose)
    # scheduler.step(J_val)


    if save:
        # Create checkpoint and save the model
        checkpoint = {
            'total_epochs': epoch,
            'model': xfoil_net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        # Save the model twice: once on its own and once in the latest model file
        torch.save(checkpoint, f'checkpoints/xfoil_net_Epoch_{epoch}_Jval_{J_val:.3e}.pth')
        torch.save(checkpoint, f'checkpoints/latest.pth')


print('Finished Training!')


# Evaluate the model
xfoil_net.eval()
X, Y = next(iter(train_dataloader))
Y_pred = xfoil_net(X)
print(torch.hstack([Y, Y_pred]))