import numpy as np
import torch
from torch import nn
from net_def import NeuralNetwork, ResNet


## Get the data
train_filepath = '../data/generated_airfoils/train/airfoil_data.npz'
dev_filepath = '../data/generated_airfoils/dev/airfoil_data.npz'
test_filepath = '../data/generated_airfoils/test/airfoil_data.npz'


data_train = np.load(train_filepath)
data_dev = np.load(dev_filepath)
data_test = np.load(test_filepath)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train = torch.from_numpy(data_train['P']).to(torch.float32).to(device)
Y_train = torch.from_numpy(data_train['L_by_D']).to(torch.float32).reshape(-1, 1).to(device)
X_val = torch.from_numpy(data_dev['P']).to(torch.float32).to(device)
Y_val = torch.from_numpy(data_dev['L_by_D']).to(torch.float32).reshape(-1, 1).to(device)
X_test = torch.from_numpy(data_test['P']).to(torch.float32).to(device)
Y_test = torch.from_numpy(data_test['L_by_D']).to(torch.float32).reshape(-1, 1).to(device)


## Initialize the network
# Set network properties
input_dim, hidden_dim, layer_count = 24, 30, 4
xfoil_net = NeuralNetwork(input_dim, hidden_dim, layer_count).to(device)
# Load saved model
checkpoint = torch.load('checkpoints/latest.pth')
xfoil_net.load_state_dict(checkpoint['model'])


## Define the loss function
MSELoss_fn = nn.MSELoss()


# Evaluate the model on training data
xfoil_net.train()
X, Y = X_train, Y_train
Y_pred = xfoil_net(X)
loss = MSELoss_fn(Y_pred, Y)
print(f'Training Loss: {loss.item()}')

# Evaluate the model on validation data
xfoil_net.eval()
X, Y = X_val, Y_val
Y_pred = xfoil_net(X)
loss = MSELoss_fn(Y_pred, Y)
print(f'Validation Loss: {loss.item()}')

# Evaluate the model on test data
xfoil_net.eval()
X, Y = X_test, Y_test
Y_pred = xfoil_net(X)
loss = MSELoss_fn(Y_pred, Y)
print(f'Test Loss: {loss.item()}')