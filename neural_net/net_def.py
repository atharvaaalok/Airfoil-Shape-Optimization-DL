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


class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_count):
        super().__init__()

        # Make sure that the layer count is an even number
        assert layer_count % 2 == 0, \
            "Layer count must be an even number as each residual block counts as 2 layers."

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        num_residual_blocks = layer_count // 2
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)]
        )

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.actf1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.actf2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.actf1(x)
        x = self.bn1(x)
        x = self.linear2(x)
        x = self.actf2(x)
        x = self.bn2(x)
        x = x + residual
        return x