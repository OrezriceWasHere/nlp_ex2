from data.word_embedder_tagger_dataset import WordEmbedderTaggerDataset
from hyper_parameters import *
from network import *
import sys
import torch
from data.dataset_helpers import generate_texts_labels
from predict import predict

import argparse

print(f'running on {DEVICE}')


def create_index_to_embedding_dict(word_to_index, word_to_embedding) -> dict:
    index_to_embedding_dict = {}

    index_to_embedding_dict[0] = word_to_embedding['UUUNKKK']
    for word, index in word_to_index.items():
        if word in word_to_embedding:
            index_to_embedding_dict[index] = word_to_embedding[word]
    return index_to_embedding_dict


def create_embedding(option, word_to_embedding, word_count, pre_count, suf_count, char_count):
    regular = SimpleEmbedding(word_count, word_to_embedding)
    presuf = PresufEmbedding(pre_count, suf_count)
    char = CharEmbedding(char_count)
    od = {'a': [regular], 'b': [char], 'c': [regular, presuf], 'd': [regular, char]}
    return Embedding(od[option])


def process_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_filename')
    parser.add_argument('task', choices=['ner', 'pos'], help='Please put the files in data/<task> dir!')
    parser.add_argument('option', choices=['a', 'b', 'c', 'd'])

    args = parser.parse_args()
    return args.task, args.model_filename, args.option


if __name__ == "__main__":
    task, model_filename, option = process_arguments()

    train_file_path = f"data/{task}/train"
    test_file_path = f"data/{task}/dev"

    class_to_index = NER_CLASS_TO_INDEX if task == 'ner' else POS_CLASS_TO_INDEX

    ignore_first = task == 'ner'
    output_length = len(NER_CLASS_TO_INDEX) if task == 'ner' else len(POS_CLASS_TO_INDEX)

    # Load word embeddings
    word_to_embedding_dict = {}
    with open("data/embedding/vocab.txt") as vocab_file, open("data/embedding/wordVectors.txt") as vector_file:
        for word, embedding in zip(vocab_file, vector_file):
            word_to_embedding_dict[word.rstrip()] = [float(x) for x in embedding.rstrip().split(" ")]

    # Prepare dataset
    train_dataset = WordEmbedderTaggerDataset(train_file_path, class_to_index)
    test_dataset = WordEmbedderTaggerDataset(test_file_path, class_to_index, train_dataset.word_to_index,
                                             train_dataset.pre_to_index, train_dataset.suf_to_index,
                                             train_dataset.char_to_index,
                                             prob_replace_to_no_word=0.0)

    index_to_embedding_dict = None

    model = Network(create_embedding(option, index_to_embedding_dict, len(train_dataset.word_to_index),
                                     len(train_dataset.pre_to_index), len(train_dataset.suf_to_index),
                                     len(train_dataset.char_to_index)), output_length)

    torch.load(model_filename)

    model = model.to(DEVICE)

    texts = list(
        generate_texts_labels(f"data/{task}/test", train_dataset.word_to_index, train_dataset.pre_to_index,
                              train_dataset.suf_to_index, train_dataset.char_to_index, None, train_dataset.dont_include,
                              False))

    index_to_class = {v: k for k, v in class_to_index.items()}

    with open(f'{task}.txt', 'w') as file:
        file.write('\n'.join([index_to_class[p] for p in predict(texts, model)]))
