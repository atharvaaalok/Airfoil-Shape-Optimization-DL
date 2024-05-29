import torch

from neural_net.net_def import NeuralNetwork


# Instantiate network
my_net = NeuralNetwork(10, 4)

m = 2
X = torch.rand(m, 10)

Y = my_net(X)
print(Y)