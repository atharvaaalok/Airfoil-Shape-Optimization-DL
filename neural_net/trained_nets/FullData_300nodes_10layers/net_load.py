import os.path

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_count):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
        for i in range(layer_count):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))

        self.linear_relu_stack = nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x



## Initialize the network
# Set network properties
input_dim, hidden_dim, layer_count = 24, 300, 10
xfoil_net = NeuralNetwork(input_dim, hidden_dim, layer_count)


# Load the model state
# Get the current scripts directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Change \ to / on windows
current_dir = current_dir.replace('\\', '/')

filename = current_dir + '/latest.pth'
device = 'cpu'
checkpoint = torch.load(filename, map_location = torch.device(device))
xfoil_net.load_state_dict(checkpoint['model'])


## Training properties
# 1. Trained on original, original_LV, original_MV, original_HV, original_MV_LV, original_HV_LV, original_HV_MV
# 2. learning_rate = 0.01
# 3. weight_decay = 1e-4
# 4. compute_optimizations = False
# 5. learning rate scheduler = ReduceLROnPlateau, factor = 0.99