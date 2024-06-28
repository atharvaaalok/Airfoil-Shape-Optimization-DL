import numpy as np
import torch
from torch import nn

from trained_nets.Final.net_load import xfoil_net, models
from utils import red, color_end


## Get the data
train_filepath = '../data/generated_airfoils/train/original.npz'
dev_filepath = '../data/generated_airfoils/dev/original.npz'
test_filepath = '../data/generated_airfoils/test/original.npz'


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
xfoil_net = xfoil_net.to(device)


## Define the loss function
MSELoss_fn = nn.MSELoss()


# Evaluate the model on training data
xfoil_net.eval()
X, Y = X_train, Y_train
Y_pred = xfoil_net(X)
loss_train = MSELoss_fn(Y_pred, Y).item()



# Evaluate the model on validation data
xfoil_net.eval()
X, Y = X_val, Y_val
Y_pred = xfoil_net(X)
loss_val = MSELoss_fn(Y_pred, Y).item()


m = 10
models = models[:m]
Y_pred_list = []
for model in models:
    model.eval()
    model.to(device)
    X, Y = X_val, Y_val
    Y_pred = model(X)
    Y_pred_list.append(Y_pred)

Y_pred = 0
for pred in Y_pred_list:
    Y_pred = Y_pred + pred
Y_pred = Y_pred / m
loss = MSELoss_fn(Y_pred, Y).item()
print(loss)

val = torch.abs(Y_pred - Y) / torch.abs(Y)
print(torch.mean(val) * 100)


# Evaluate the model on test data
xfoil_net.eval()
X, Y = X_test, Y_test
Y_pred = xfoil_net(X)
loss_test = MSELoss_fn(Y_pred, Y).item()


models = models[:m]
Y_pred_list = []
for model in models:
    model.eval()
    model.to(device)
    X, Y = X_test, Y_test
    Y_pred = model(X)
    Y_pred_list.append(Y_pred)

Y_pred = 0
for pred in Y_pred_list:
    Y_pred = Y_pred + pred
Y_pred = Y_pred / m
loss = MSELoss_fn(Y_pred, Y).item()
print(loss)

# Print the losses
print(f'{red}Training Loss   :{color_end} {loss_train:10.6f}')
print(f'{red}Validation Loss :{color_end} {loss_val:10.6f}')
print(f'{red}Test Loss       :{color_end} {loss_test:10.6f}')