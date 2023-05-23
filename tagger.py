from data.word_embedder_tagger_dataset import WordEmbedderTaggerDataset
from hyper_parameters import *
from network import *
from trainer import train
import sys
import torch
from data.dataset_helpers import generate_texts_labels
from predict import predict

from interpretability import explain

print(f'running on {DEVICE}')


def create_index_to_embedding_dict(word_to_index, word_to_embedding) -> dict:
    index_to_embedding_dict = {}

    index_to_embedding_dict[2] = word_to_embedding['UUUNKKK']
    for word, index in word_to_index.items():
        if word in word_to_embedding:
            index_to_embedding_dict[index] = word_to_embedding[word]
    return index_to_embedding_dict


if __name__ == "__main__":
    presuf = sys.argv[3] == 'presuf'
    chars = sys.argv[3] == 'chars'

    task = sys.argv[1].lower()

    train_file_path = f"data/{task}/train"
    test_file_path = f"data/{task}/dev"

    class_to_index = NER_CLASS_TO_INDEX if task == 'ner' else POS_CLASS_TO_INDEX

    epochs = EPOCHS if task == 'ner' else EPOCHS // 2

    if task == 'ner' and chars:
        layers = CHAR_NER_LAYERS
    elif task == 'ner':
        layers = NER_LAYERS
    elif chars:
        layers = CHAR_POS_LAYERS
    else:
        layers = POS_LAYERS

    ignore_first = task == 'ner'

    # Load word embeddings
    word_to_embedding_dict = {}
    with open("data/embedding/vocab.txt") as vocab_file, open("data/embedding/wordVectors.txt") as vector_file:
        for word, embedding in zip(vocab_file, vector_file):
            word_to_embedding_dict[word.rstrip()] = [float(x) for x in embedding.rstrip().split(" ")]

    # Prepare dataset
    train_dataset = WordEmbedderTaggerDataset(presuf, chars, train_file_path, class_to_index)
    test_dataset = WordEmbedderTaggerDataset(presuf, chars, test_file_path, class_to_index, train_dataset.word_to_index,
                                             train_dataset.pre_to_index, train_dataset.suf_to_index,
                                             train_dataset.char_to_index,
                                             prob_replace_to_no_word=0.0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    index_to_embedding_dict = None
    if sys.argv[2] == 'pre':
        index_to_embedding_dict = create_index_to_embedding_dict(train_dataset.word_to_index, word_to_embedding_dict)
        print('with pre-trained embedding!')
    else:
        print('without pre-trained embedding!')

    # Prepare model
    if presuf:
        model = WithPresufEmbedding(len(train_dataset.pre_to_index), len(train_dataset.suf_to_index),
                                    len(train_dataset.word_to_index), layers, index_to_embedding_dict)
    elif chars:
        model = WithCharacterEmbedding(len(train_dataset.char_to_index), len(train_dataset.word_to_index), layers,
                                       index_to_embedding_dict)
    else:
        model = SequenceTaggerWithEmbeddingNetwork(len(train_dataset.word_to_index), layers,
                                                   index_to_embedding_dict)

    model = model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    if task == 'ner' and chars:
        criterion = torch.nn.CrossEntropyLoss(
            torch.tensor([0.05, 0.225, 0.225, 0.225, 0.225]).to(DEVICE))  # don't panic! it will changed later!

    train(train_loader, test_loader, model, criterion, optimizer, f'{task}_{len(sys.argv)}', epochs, ignore_first)

    texts = list(
        generate_texts_labels(f"data/{task}/test", train_dataset.word_to_index, train_dataset.pre_to_index,
                              train_dataset.suf_to_index, train_dataset.char_to_index, None, train_dataset.dont_include,
                              False, presuf, chars))
    # test_texts = torch.tensor(texts, dtype=torch.int).to(DEVICE)

    index_to_class = {v: k for k, v in class_to_index.items()}

    with open(f'{task}_{len(sys.argv)}.txt', 'w') as file:
        file.write('\n'.join([index_to_class[p] for p in predict(texts, model, chars)]))

    if chars:
        explain(model, list(train_dataset.pre_to_index.keys()) + list(train_dataset.suf_to_index.keys()),
                train_dataset.char_to_index)
