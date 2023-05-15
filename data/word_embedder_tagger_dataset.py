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

    def __init__(self, presuf, tagged_file, tag_to_index, word_to_index=None, pre_to_index=None, suf_to_index=None,
                 prob_replace_to_no_word=PROB_UNQ):
        self.word_to_index, self.pre_to_index, self.suf_to_index = \
            (word_to_index, pre_to_index,
             suf_to_index) if (word_to_index is not None) else dataset_helpers.word_to_index_dict(tagged_file)

        self.no_word = self.word_to_index[NO_WORD]
        self.no_pre = self.pre_to_index[NO_WORD]
        self.no_suf = self.suf_to_index[NO_WORD]
        self.tag_to_index = tag_to_index

        self.dont_include = []

        texts_labels = list(dataset_helpers.generate_texts_labels(tagged_file,
                                                                  self.word_to_index,
                                                                  self.pre_to_index,
                                                                  self.suf_to_index,
                                                                  self.tag_to_index,
                                                                  self.dont_include,
                                                                  True))

        texts, labels = [], []

        for _ in range(AUGMENTATION_COUNT):
            for text, pre, suf, label in texts_labels:
                for i in range(len(text)):
                    if text[i] > 2 and torch.rand(1) < prob_replace_to_no_word:
                        text[i] = self.no_word
                        if torch.rand(1) < 0.15:
                            pre[i] = self.no_pre
                        if torch.rand(1) < 0.15:
                            suf[i] = self.no_suf

                if presuf:
                    texts.append((text, pre, suf))
                else:
                    texts.append(text)
                labels.append(label)

        self.texts = torch.tensor(texts, dtype=torch.int).to(DEVICE)

        self.labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
