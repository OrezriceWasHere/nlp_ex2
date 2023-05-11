import torch
import itertools
from hyper_parameters import *


class WordTaggerDataset(torch.utils.data.Dataset):
    """
    A class that represents a dataset of tagged words.
    Words are in an order like this:
    Hello A
    I B
    Am C
    Or D
    Shachar E
    <End of line>

    The dataset will return a record of some words and their tag, according to the window size.
    For example, if the window size is 5, the dataset will return:
    Hello I Am Or Shachar -> 2
    I Am Or Shachar <END_PAD> -> 3
    The words are embedded according to the word_to_embedding_dict.
    """

    def __init__(self, tagged_file, tag_to_index, word_to_embedding_dict, prob_replace_to_no_word=PROB_UNQ):
        torch.seed()
        self.texts = []
        self.labels = []
        start_embedding = word_to_embedding_dict[START_PAD]
        end_embedding = word_to_embedding_dict[END_PAD]
        start_buffer = [start_embedding, start_embedding, end_embedding, end_embedding]
        buffer = start_buffer.copy()
        labels = []
        start_index = WINDOW // 2 + (1 - (WINDOW % 2))
        current_index = start_index
        with open(tagged_file) as f:

            for line in f:

                # A sentence is ended
                if line == "\n":
                    for i in range(len(buffer) - (WINDOW - 1)):
                        # Convert a list of embedding of 5 words to a single vector
                        self.texts.append(buffer[i:i + WINDOW])
                        self.labels.append(labels[i])

                    buffer = start_buffer.copy()
                    labels = []
                    current_index = start_index
                    continue

                word, tag = line.rstrip().split("\t")
                if word.lower() in word_to_embedding_dict:
                    if torch.rand(1) < prob_replace_to_no_word:
                        word = NO_WORD
                embedding = word_to_embedding_dict.get(word.lower()) or word_to_embedding_dict[NO_WORD]
                buffer.insert(current_index, embedding)
                current_index += 1
                labels.append(tag_to_index[tag])


    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_label_at(self, idx):
        return self.labels[idx]

    def get_text_at(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        text = torch.tensor(self.get_text_at(idx), dtype=torch.float)
        label = torch.tensor(self.get_label_at(idx), dtype=torch.long)
        return text, label
