import os.path
import time

import numpy as np
import torch
from torch import nn

from net_def import NeuralNetwork, ResNet
from utils import train_loop, dev_loop, red, cyan, color_end


## Get the data
train_filepath = '../data/generated_airfoils/train/airfoil_data.npz'
dev_filepath = '../data/generated_airfoils/dev/airfoil_data.npz'

data_train = np.load(train_filepath)
data_dev = np.load(dev_filepath)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train = torch.from_numpy(data_train['P']).to(torch.float32).to(device)
Y_train = torch.from_numpy(data_train['L_by_D']).to(torch.float32).reshape(-1, 1).to(device)
X_val = torch.from_numpy(data_dev['P']).to(torch.float32).to(device)
Y_val = torch.from_numpy(data_dev['L_by_D']).to(torch.float32).reshape(-1, 1).to(device)


## Initialize the network
# Set network properties
input_dim, hidden_dim, layer_count = 24, 30, 4
xfoil_net = NeuralNetwork(input_dim, hidden_dim, layer_count).to(device)


# Make changes for running the computation faster
compute_optimizations = False
if compute_optimizations == True:
    try:
        xfoil_net = torch.compile(xfoil_net)
    except:
        print('Could not compile the network.')
    torch.set_float32_matmul_precision('high')


## Define the loss function
MSELoss_fn = nn.MSELoss()


## Define an optimizer
learning_rate = 0.01
weight_decay = 0
optimizer = torch.optim.Adam(xfoil_net.parameters(), lr = learning_rate, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)


## Train the network
# Set the training properties
epochs = 1000000
print_cost_every = 1000
B_train = X_train.shape[0]
B_dev = X_val.shape[0]


# Load saved model if available
if os.path.exists('checkpoints/latest.pth'):
    checkpoint = torch.load('checkpoints/latest.pth')
    xfoil_net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    total_epochs = checkpoint['total_epochs']
else:
    total_epochs = 0


# Train the network
for epoch in range(total_epochs + 1, total_epochs + epochs + 1):

    verbose = True if epoch % print_cost_every == 0 or epoch == total_epochs + 1 else False
    save = verbose

    if verbose:
        print(f'{red}Epoch {epoch}{color_end}\n' + 40 * '-')
        print(scheduler.get_last_lr())
        t0 = time.perf_counter()
        
    

    # Run the training loop
    J_train = train_loop(X_train, Y_train, B_train, xfoil_net, MSELoss_fn, optimizer, verbose = verbose, compute_optimizations = compute_optimizations)

    # Run the validation loop
    J_val = dev_loop(X_val, Y_val, B_dev, xfoil_net, MSELoss_fn, verbose, compute_optimizations = compute_optimizations)
    scheduler.step(J_val)


    if verbose:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000 # Time difference in milliseconds
        print(f'Time taken: {dt}')


    if save:
        # Create checkpoint and save the model
        checkpoint = {
            'total_epochs': epoch,
            'model': xfoil_net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        # Save the model twice: once on its own and once in the latest model file
        torch.save(checkpoint, f'checkpoints/xfoil_net_Epoch_{epoch}_Jtrain{J_train:.3e}_Jval_{J_val:.3e}.pth')
        torch.save(checkpoint, f'checkpoints/latest.pth')


print('Finished Training!')