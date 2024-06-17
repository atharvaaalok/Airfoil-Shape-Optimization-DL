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


class Model_Original_TrainError0(nn.Module):
    
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