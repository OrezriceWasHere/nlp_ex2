import torch
from hyper_parameters import *
from trainer import train
from data.word_embedder_tagger_dataset import WordEmbedderTaggerDataset
from sequence_tagger_with_embedding_network import SequenceTaggerWithEmbeddingNetwork
import sys


def create_index_to_embedding_dict(word_to_index, word_to_embedding) -> dict:
    index_to_embedding_dict = {}
    for word, index in word_to_index.items():
        if word in word_to_embedding:
            index_to_embedding_dict[index] = word_to_embedding[word]
    return index_to_embedding_dict


if __name__ == "__main__":
    ner_train_file_path = f"data/{sys.argv[1]}/train"
    ner_test_file_path = f"data/{sys.argv[1]}/dev"

    # Load word embeddings
    word_to_embedding_dict = {}
    with open("data/embedding/vocab.txt") as vocab_file, open("data/embedding/wordVectors.txt") as vector_file:
        for word, embedding in zip(vocab_file, vector_file):
            word_to_embedding_dict[word.rstrip()] = [float(x) for x in embedding.rstrip().split(" ")]

    # Preprare dataset
    train_dataset = WordEmbedderTaggerDataset(ner_train_file_path, NER_CLASS_TO_INDEX)
    word_to_index = train_dataset.word_to_index
    test_dataset = WordEmbedderTaggerDataset(ner_test_file_path, NER_CLASS_TO_INDEX,
                                             word_to_index=word_to_index,
                                             prob_replace_to_no_word=0.0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    index_to_embedding_dict = create_index_to_embedding_dict(word_to_index, word_to_embedding_dict)

    # Prepare model
    model = SequenceTaggerWithEmbeddingNetwork(len(train_dataset.word_to_index),
                                               NER_LAYERS,
                                               index_to_embedding=index_to_embedding_dict)
    model = model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(train_loader, test_loader, model, criterion, optimizer)
