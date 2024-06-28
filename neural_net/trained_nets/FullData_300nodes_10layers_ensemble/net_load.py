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


filenames = ['xfoil_net_Epoch_2224_Jtrain8.185e-02_Jval_9.928e-01.pth',
             'xfoil_net_Epoch_1846_Jtrain1.184e-01_Jval_1.094e+00.pth',
             'xfoil_net_Epoch_1767_Jtrain1.225e-01_Jval_1.102e+00.pth',
             'xfoil_net_Epoch_2052_Jtrain1.070e-01_Jval_1.177e+00.pth',
             'xfoil_net_Epoch_2201_Jtrain8.780e-02_Jval_1.180e+00.pth',
             'xfoil_net_Epoch_2049_Jtrain1.117e-01_Jval_1.182e+00.pth',
             'xfoil_net_Epoch_1821_Jtrain1.364e-01_Jval_1.188e+00.pth',
             'xfoil_net_Epoch_1815_Jtrain1.366e-01_Jval_1.206e+00.pth',
             'xfoil_net_Epoch_2003_Jtrain1.449e-01_Jval_1.208e+00.pth',
             'xfoil_net_Epoch_1552_Jtrain1.608e-01_Jval_1.220e+00.pth']



# Load the model state
# Get the current scripts directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Change \ to / on windows
current_dir = current_dir.replace('\\', '/')

# Get the checkpoints
checkpoint_list = []
for name in filenames:
    file_path = current_dir + '/' + name
    checkpoint_list.append(torch.load(file_path))

# Get the state dict for the model from each checkpoint
state_dict_list = []
for checkpoint in checkpoint_list:
    state_dict = checkpoint['model']
    state_dict_list.append(state_dict)


# Create average of the state dicts
state_dict_avg = state_dict_list[0]

n = 1
for key in state_dict_avg:
    for state_dict in state_dict_list[1:n]:
        state_dict_avg[key] = state_dict_avg[key] + state_dict[key]
    
    # Find average of the values
    state_dict_avg[key] = state_dict_avg[key] / n



xfoil_net.load_state_dict(state_dict_avg)

# Create models for all the checkpoints
models = [NeuralNetwork(input_dim, hidden_dim, layer_count) for _ in range(len(filenames))]
for i, state_dict in enumerate(state_dict_list):
    models[i].load_state_dict(state_dict)


## Training properties
# 1. Trained on original, original_LV, original_MV, original_HV, original_MV_LV, original_HV_LV, original_HV_MV
# 2. learning_rate = 0.01
# 3. weight_decay = 1e-4
# 4. compute_optimizations = False
# 5. learning rate scheduler = ReduceLROnPlateau, factor = 0.99