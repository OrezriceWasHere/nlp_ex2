import torch
from hyper_parameters import *


class SequenceTaggerWithEmbeddingNetwork(torch.nn.Module):

    def __init__(self,
                 count_distinct_words,
                 network_structure,
                 dropout=DROPOUT,
                 index_to_embedding=None
                 ):
        super(SequenceTaggerWithEmbeddingNetwork, self).__init__()
        self.network_structure = network_structure
        self.embedding = self.__create_embedding_layer(count_distinct_words, EMBEDDING_SIZE, index_to_embedding)
        layers = self.__prepare_network_layers(dropout, network_structure)
        self.net = torch.nn.Sequential(*layers)

    def __create_embedding_layer(self, distinct_words, embedding_size, index_to_embedding_dict):
        if index_to_embedding_dict is None:
            return torch.nn.Embedding(distinct_words, embedding_size)
        else:
            embedding_matrix = torch.zeros(distinct_words, embedding_size)
            for index, embedding in index_to_embedding_dict.items():
                embedding_matrix[index] = torch.tensor(embedding)
            return torch.nn.Embedding.from_pretrained(embedding_matrix)

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
