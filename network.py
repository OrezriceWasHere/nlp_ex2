import torch
from hyper_parameters import *
import math


class SequenceTaggerWithEmbeddingNetwork(torch.nn.Module):

    def __init__(self,
                 count_distinct_words,
                 network_structure,
                 index_to_embedding=None
                 ):
        super().__init__()
        self.network_structure = network_structure
        self.embedding = self.__create_embedding_layer(count_distinct_words, EMBEDDING_SIZE, index_to_embedding)
        layers = self.__prepare_network_layers(DROPOUT, network_structure)
        self.net = torch.nn.Sequential(*layers)

    def __create_embedding_layer(self, distinct_words, embedding_size, index_to_embedding_dict):
        if index_to_embedding_dict is None:
            return torch.nn.Embedding(distinct_words, embedding_size)
        else:
            embedding_matrix = torch.tensor(
                torch.nn.Embedding(distinct_words, embedding_size).weight)
            for index, embedding in index_to_embedding_dict.items():
                embedding_matrix[index] = torch.tensor(embedding)
            return torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

    def __prepare_network_layers(self, dropout, network_structure):
        inner = network_structure[1:-1]

        layers = [torch.nn.Dropout(dropout), torch.nn.Linear(network_structure[0], network_structure[1]),
                  torch.nn.Tanh()]

        for pre_index, post_index in zip(inner, inner[1:]):
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(pre_index, post_index))
            layers.append(torch.nn.LeakyReLU())

        layers.extend([torch.nn.Linear(network_structure[-2], network_structure[-1]), torch.nn.Softmax(dim=1)])
        return layers

    def embed(self, *x):
        return self.embedding(*x).view(-1, WINDOW * EMBEDDING_SIZE)

    def forward(self, *x):
        x = self.embed(*x)
        return self.net(x)


class WithPresufEmbedding(SequenceTaggerWithEmbeddingNetwork):
    def __init__(self, pre, suf, *args):
        super().__init__(*args)
        self.pre_embedding = torch.nn.Embedding(pre, EMBEDDING_SIZE)
        self.suf_embedding = torch.nn.Embedding(suf, EMBEDDING_SIZE)

    def embed(self, x):
        return super().embed(x[:, 0, :]) + \
               self.pre_embedding(x[:, 1, :]).view(-1, WINDOW * EMBEDDING_SIZE) + \
               self.suf_embedding(x[:, 2, :]).view(-1, WINDOW * EMBEDDING_SIZE)


class WithCharacterEmbedding(SequenceTaggerWithEmbeddingNetwork):
    def __init__(self, characters_number, *args):
        super().__init__(*args)
        self.char_embedding = torch.nn.Embedding(characters_number, CHARACTER_EMBEDDING_SIZE)
        self.char_embedding.weight.data.uniform_(-math.sqrt(3 / CHARACTER_EMBEDDING_SIZE),
                                                 math.sqrt(3 / CHARACTER_EMBEDDING_SIZE))
        self.conv_ = torch.nn.Conv3d(1, FILTERS_NUMBER, (1, FILTER_SIZE, CHARACTER_EMBEDDING_SIZE)).to(DEVICE)
        self.conv = torch.nn.Sequential(torch.nn.Dropout(0.2), self.conv_)
        self.pool = torch.nn.AdaptiveMaxPool1d(1)

    def embed(self, words, characters):
        return torch.cat((super().embed(words), self.embed_characters(characters)), 1)

    def embed_characters(self, characters):
        embedded = self.char_embedding(characters)
        reshaped = embedded.unsqueeze(1)
        conved = self.conv(reshaped)
        reshaped = conved.view(conved.size(0), conved.size(1) * conved.size(2), conved.size(3))
        pooled = self.pool(reshaped)
        return pooled.view(pooled.size(0), pooled.size(1))
