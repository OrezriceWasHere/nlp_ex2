import torch
from hyper_parameters import *


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
            embedding_matrix = torch.zeros(distinct_words, embedding_size)
            for index, embedding in index_to_embedding_dict.items():
                embedding_matrix[index] = torch.tensor(embedding)
            return torch.nn.Embedding.from_pretrained(embedding_matrix)

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
        self.conv = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Conv2d(WINDOW, FILTERS_NUMBER, FILTERS_SIZE),
                                        torch.nn.MaxPool2d(kernel_size=(MAX_CHARACTERS - (FILTERS_SIZE - 1), 1),
                                                           stride=(MAX_CHARACTERS - (FILTERS_SIZE - 1), 1))
                                        )

    def embed(self, words, characters):
        embedded = self.char_embedding(characters)
        conved = self.conv(embedded)
        reshaped = conved.view(conved.size(0), conved.size(1) * conved.size(2) * conved.size(3))
        return torch.cat((super().embed(words), reshaped), 1)

    def embed_characters(self, x):
        self.char_embedding(x)
