from data.word_embedder_tagger_dataset import WordEmbedderTaggerDataset
from hyper_parameters import *
from sequence_tagger_with_embedding_network import SequenceTaggerWithEmbeddingNetwork
from trainer import train
import sys
import torch
from data.dataset_helpers import generate_texts
from predict import predict

print(f'running on {DEVICE}')


def create_index_to_embedding_dict(word_to_index, word_to_embedding) -> dict:
    index_to_embedding_dict = {}
    for word, index in word_to_index.items():
        if word in word_to_embedding:
            index_to_embedding_dict[index] = word_to_embedding[word]
    return index_to_embedding_dict


if __name__ == "__main__":
    task = sys.argv[1].lower()

    train_file_path = f"data/{task}/train"
    test_file_path = f"data/{task}/dev"

    class_to_index, layers = (NER_CLASS_TO_INDEX, NER_LAYERS) if task == 'ner' else (POS_CLASS_TO_INDEX, POS_LAYERS)
    ignore_first = task == 'ner'

    # Load word embeddings
    word_to_embedding_dict = {}
    with open("data/embedding/vocab.txt") as vocab_file, open("data/embedding/wordVectors.txt") as vector_file:
        for word, embedding in zip(vocab_file, vector_file):
            word_to_embedding_dict[word.rstrip()] = [float(x) for x in embedding.rstrip().split(" ")]

    # Prepare dataset
    train_dataset = WordEmbedderTaggerDataset(train_file_path, class_to_index)
    word_to_index = train_dataset.word_to_index
    test_dataset = WordEmbedderTaggerDataset(test_file_path, class_to_index,
                                             word_to_index=word_to_index,
                                             prob_replace_to_no_word=0.0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    index_to_embedding_dict = None
    if len(sys.argv) >= 3:
        index_to_embedding_dict = create_index_to_embedding_dict(word_to_index, word_to_embedding_dict)
        print('with pre-trained embedding!')
    else:
        print('without pre-trained embedding!')

    # Prepare model
    model = SequenceTaggerWithEmbeddingNetwork(len(train_dataset.word_to_index), layers,
                                               index_to_embedding=index_to_embedding_dict)
    model = model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train(train_loader, test_loader, model, criterion, optimizer, ignore_first)

    texts = list(generate_texts(f"data/{task}/test", train_dataset.word_to_index, train_dataset.dont_include))
    test_texts = torch.tensor(texts, dtype=torch.int).to(DEVICE)

    index_to_class = {v: k for k, v in class_to_index.items()}

    with open(f'{task}_{len(sys.argv)}.txt', 'w') as file:
        file.write('\n'.join([index_to_class[p] for p in predict(test_texts, model)]))
