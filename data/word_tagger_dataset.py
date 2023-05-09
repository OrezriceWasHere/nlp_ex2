import torch
import itertools
from hyper_parameters import *


class WordTaggerDataset(torch.utils.data.Dataset):

    def __init__(self, tagged_file, word_to_embedding_dict, tag_to_index):
        self.texts = []
        self.labels = []
        start_embedding = word_to_embedding_dict[START_PAD]
        end_embedding = word_to_embedding_dict[END_PAD]
        start_buffer = [start_embedding, start_embedding, end_embedding, end_embedding]
        buffer = start_buffer.copy()
        labels = []
        start_index = WINDOW // 2 + (1 - (WINDOW % 2))
        current_index = start_index
        not_found_words = set()
        with open(tagged_file) as f:

            for line in f:

                # A sentence is ended
                if line == "\n":
                    for i in range(len(buffer) - 4):
                        # Convert a list of embedding of 5 words to a single vector
                        self.texts.append(list(itertools.chain.from_iterable(buffer[i:i + WINDOW])))
                        self.labels.append(labels[i])

                    buffer = start_buffer.copy()
                    labels = []
                    current_index = start_index
                    continue

                word, tag = line.rstrip().split("\t")
                embedding = word_to_embedding_dict.get(word.lower(), NO_WORD_EMBEDDING)
                buffer.insert(current_index, embedding)
                current_index += 1
                labels.append(tag_to_index[tag])


    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_label_at(self, idx):
        return torch.tensor(self.labels[idx], dtype=torch.long)

    def get_embedding_at(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.float32)

    def __getitem__(self, idx):
        text = self.get_embedding_at(idx)
        label = self.get_label_at(idx)
        return text, label
