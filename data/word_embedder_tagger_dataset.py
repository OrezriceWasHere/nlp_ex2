import pandas as pd
import torch
from hyper_parameters import *

from data.word_tagger_dataset import WordTaggerDataset


class WordEmbedderTaggerDataset(torch.utils.data.Dataset):
    def __init__(self,tagged_file, tag_to_index, word_to_index=None, prob_replace_to_no_word=0.1):
        self.word_to_index = word_to_index
        if word_to_index is None:
            self.create_word_to_index(tagged_file)
        self.word_tagger_dataset = WordTaggerDataset(tagged_file,tag_to_index, self.word_to_index, prob_replace_to_no_word)

    def create_word_to_index(self, tagged_file):
        distinct_words = set()
        with open(tagged_file) as f:
            for line in f:
                if line == "\n":
                    continue
                word, _ = line.rstrip().split("\t")
                distinct_words.add(word.lower())
        self.word_to_index = {word: index for index, word in enumerate(distinct_words)}
        special_words = [START_PAD, END_PAD, NO_WORD]
        for offset, word in enumerate(special_words):
            self.word_to_index[word] = len(distinct_words) + offset

    def word_to_index(self):
        return self.word_to_index

    def __len__(self):
        return len(self.word_tagger_dataset)

    def __getitem__(self, idx):
        text = torch.tensor(self.word_tagger_dataset.get_text_at(idx), dtype=torch.int)
        label = torch.tensor(self.word_tagger_dataset.get_label_at(idx), dtype=torch.long)
        return text, label

