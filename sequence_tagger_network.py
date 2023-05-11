import torch
from hyper_parameters import *

class SequenceTaggerNetwork(torch.nn.Module):

    def __init__(self, network_structure, dropout=DROPOUT):
        super(SequenceTaggerNetwork, self).__init__()
        self.network_structure = network_structure
        layers = [torch.nn.Dropout(dropout)]
        for pre_index, post_index in zip(network_structure, network_structure[1:]):
            layers.append(torch.nn.Linear(pre_index, post_index))
            layers.append(torch.nn.Tanh())
        layers.pop()
        self.net = torch.nn.Sequential(*layers)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        return self.softmax(x)
