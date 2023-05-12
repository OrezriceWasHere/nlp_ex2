import torch
from hyper_parameters import *


class SequenceTaggerWithEmbeddingNetwork(torch.nn.Module):

    def __init__(self, count_distinct_words, network_structure, dropout=DROPOUT):
        super(SequenceTaggerWithEmbeddingNetwork, self).__init__()
        self.embedding = torch.nn.Embedding(count_distinct_words, EMBEDDING_SIZE)
        self.network_structure = network_structure
        layers = self.__prepare_network_layers(dropout, network_structure)
        self.net = torch.nn.Sequential(*layers)

    def __prepare_network_layers(self, dropout, network_structure):
        layers = [torch.nn.Dropout(dropout)]
        for pre_index, post_index in zip(network_structure, network_structure[1:]):
            layers.append(torch.nn.Linear(pre_index, post_index))
            layers.append(torch.nn.Tanh())
        layers.pop()
        layers.append(torch.nn.Softmax(dim=1))
        return layers

    def forward(self, x):
        x = self.embedding(x).view(-1, WINDOW * EMBEDDING_SIZE)
        return self.net(x)
