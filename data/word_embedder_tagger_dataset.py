from hyper_parameters import *
from data import dataset_helpers


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

    def __init__(self, tagged_file, tag_to_index, word_to_index=None, prob_replace_to_no_word=PROB_UNQ):
        self.word_to_index = word_to_index or dataset_helpers.word_to_index_dict(tagged_file)
        self.tag_to_index = tag_to_index

        texts_labels_generator = dataset_helpers.generate_texts_labels(tagged_file,
                                                              self.word_to_index,
                                                              self.tag_to_index,
                                                              prob_replace_to_no_word)

        self.texts, self.labels = [], []
        for text, label in texts_labels_generator:
            self.texts.append(text)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.int)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return text, label
