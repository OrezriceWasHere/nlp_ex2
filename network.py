import torch
from hyper_parameters import *


class SimpleEmbedding(torch.nn.Module):

    def __init__(self, words_count, index_to_embedding=None):
        super().__init__()

        if index_to_embedding is None:
            self.embedding = torch.nn.Embedding(words_count, EMBEDDING_SIZE)
        else:
            embedding_matrix = torch.nn.Embedding(words_count, EMBEDDING_SIZE).weight.clone().detach()
            for index, embedding in index_to_embedding.items():
                embedding_matrix[index] = torch.tensor(embedding)
            self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        self.embedding = self.embedding.to(DEVICE)

    def __len__(self):
        return EMBEDDING_SIZE

    def forward(self, words, *_):
        return self.embedding(words)


class PresufEmbedding(torch.nn.Module):

    def __init__(self, pre_count, suf_count):
        super().__init__()

        self.pre = torch.nn.Embedding(pre_count, EMBEDDING_SIZE).to(DEVICE)
        self.suf = torch.nn.Embedding(suf_count, EMBEDDING_SIZE).to(DEVICE)

    def __len__(self):
        return EMBEDDING_SIZE * 2

    def forward(self, _, pre, suf, *__):
        return torch.cat([self.pre(pre), self.suf(suf)], dim=1)


class CharEmbedding(torch.nn.Module):

    def __init__(self, chars_count):
        super().__init__()
        self.chars_embedding = torch.nn.Embedding(chars_count, CHARACTER_EMBEDDING_SIZE).to(DEVICE)
        self.lstm = torch.nn.LSTM(CHARACTER_EMBEDDING_SIZE, CHAR_LSTM_HIDDEN_SIZE).to(DEVICE)

    def __len__(self):
        return CHAR_LSTM_HIDDEN_SIZE

    def forward(self, _, __, ___, chars_s):
        zeros = torch.zeros(1, CHAR_LSTM_HIDDEN_SIZE).to(DEVICE)
        result = torch.zeros(len(chars_s), CHAR_LSTM_HIDDEN_SIZE).to(DEVICE)
        for i, chars in enumerate(chars_s):
            out, _ = self.lstm(self.chars_embedding(chars), (zeros, zeros))
            result[i] = out[-1, :]
        return result


class Embedding(torch.nn.Module):

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def __len__(self):
        return sum(len(e) for e in self.embeddings)

    def forward(self, *args):
        return torch.cat([e(*args) for e in self.embeddings], dim=1)


class Network(torch.nn.Module):

    def __init__(self, embedding, output_length):
        super().__init__()
        self.embedding = embedding

        self.bilstm = torch.nn.LSTM(len(embedding), LSTM_HIDDEN_SIZE, num_layers=2, bidirectional=True, dropout=DROPOUT)

        self.classifier = torch.nn.Sequential(torch.nn.Linear(LSTM_HIDDEN_SIZE * 2, output_length), torch.nn.Softmax(0))

    def forward(self, *args):
        x = self.embedding(*args)
        output, _ = self.bilstm(x)
        return self.classifier(output)
