import torch
from hyper_parameters import *
from sequence_tagger_network import SequenceTaggerNetwork

class SequenceTaggerWithEmbeddingNetwork(torch.nn.Module):

    def __init__(self, count_distinct_words, network_structure):
        super(SequenceTaggerWithEmbeddingNetwork, self).__init__()
        self.embedding = torch.nn.Embedding(count_distinct_words, EMBEDDING_SIZE)
        self.net = SequenceTaggerNetwork(network_structure)

    def forward(self, x):
        x = self.embedding(x).view(-1, WINDOW * EMBEDDING_SIZE)
        return self.net(x)
