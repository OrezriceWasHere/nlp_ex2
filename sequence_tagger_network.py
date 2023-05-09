import torch


class SequenceTaggerNetwork(torch.nn.Module):

    def __init__(self, network_structure):
        super(SequenceTaggerNetwork, self).__init__()
        self.network_structure = network_structure
        layers = []
        for pre_index, post_index in zip(network_structure, network_structure[1:]):
            layers.append(torch.nn.Linear(pre_index, post_index))
            layers.append(torch.nn.Tanh())
        layers.pop()
        layers.append(torch.nn.Softmax(dim=1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
