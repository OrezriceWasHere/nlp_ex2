from hyper_parameters import *
from data import dataset_helpers
import torch


class WordEmbedderTaggerDataset(torch.utils.data.Dataset):
    """
    A class that represents a dataset of tagged words.
    Words are in an order like this:
    Hello A
    I B
    Am C
    Or D
    Shachar E
    <End of line>

    tags are in a mapping like this:
    A -> 0
    B -> 1
    C -> 2
    D -> 3
    E -> 4

    words are in a mapping like this:
    <s> -> 0
    </s> -> 1
    <unk> -> 2
    Hello -> 3
    I -> 4
    Am -> 5
    Or -> 6
    Shachar -> 7


    The dataset will return a record of some words and their tag, according to the window size.
    For example, if the window size is 5, the dataset will return:
    [0, 0, 3, 4, 5], 0
    [0, 3, 4, 5, 6], 1
    [3, 4, 5, 6, 7], 2
    [4, 5, 6, 7, 1], 3
    [5, 6, 7, 1, 1], 4
    ...

    The words are embedded according to the word_to_embedding_dict.
    """

    def __init__(self, tagged_file, tag_to_index, word_to_index=None, pre_to_index=None,
                 suf_to_index=None,
                 char_to_index=None, prob_replace_to_no_word=PROB_UNQ):
        self.word_to_index, self.pre_to_index, self.suf_to_index, self.char_to_index = \
            (word_to_index, pre_to_index,
             suf_to_index, char_to_index) if (word_to_index is not None) else dataset_helpers.word_to_index_dict(
                tagged_file)

        self.no_word = self.word_to_index[NO_WORD]
        self.no_pre = self.pre_to_index[NO_WORD]
        self.no_suf = self.suf_to_index[NO_WORD]
        self.tag_to_index = tag_to_index

        self.dont_include = []

        texts_labels = list(dataset_helpers.generate_texts_labels(tagged_file,
                                                                  self.word_to_index,
                                                                  self.pre_to_index,
                                                                  self.suf_to_index,
                                                                  self.char_to_index,
                                                                  self.tag_to_index,
                                                                  self.dont_include,
                                                                  True))

        self.texts, self.prefixes, self.suffixes, self.labels = [], [], [], []
        self.chars = []

        for text, pre, suf, cha, label in texts_labels:
            for i in range(len(text)):
                if torch.rand(1) < prob_replace_to_no_word:
                    text[i] = self.no_word
                    if torch.rand(1) < 0.15:
                        pre[i] = self.no_pre
                    if torch.rand(1) < 0.15:
                        suf[i] = self.no_suf

            self.texts.append(torch.tensor(text, dtype=torch.int).to(DEVICE))
            self.prefixes.append(torch.tensor(pre, dtype=torch.int).to(DEVICE))
            self.suffixes.append(torch.tensor(suf, dtype=torch.int).to(DEVICE))
            self.labels.append(torch.tensor(label, dtype=torch.long).to(DEVICE))
            self.chars.append([torch.tensor(c, dtype=torch.int).to(DEVICE) for c in cha])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.texts[idx], self.prefixes[idx], self.suffixes[idx], self.chars[idx]), self.labels[idx]
