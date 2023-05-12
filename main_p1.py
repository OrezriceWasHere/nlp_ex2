from data.word_embedder_tagger_dataset import WordEmbedderTaggerDataset
from hyper_parameters import *
from sequence_tagger_with_embedding_network import SequenceTaggerWithEmbeddingNetwork
from trainer import train

print(f'running on {DEVICE}')

if __name__ == "__main__":
    ner_train_file_path = "data/ner/train"
    ner_test_file_path = "data/ner/dev"


    # Preprare dataset
    train_dataset = WordEmbedderTaggerDataset(ner_train_file_path, NER_CLASS_TO_INDEX)
    word_to_index = train_dataset.word_to_index
    test_dataset = WordEmbedderTaggerDataset(ner_test_file_path, NER_CLASS_TO_INDEX,
                                             word_to_index=word_to_index,
                                             prob_replace_to_no_word=0.0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Prepare model
    model = SequenceTaggerWithEmbeddingNetwork(len(train_dataset.word_to_index), NER_LAYERS)
    model = model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(train_loader, test_loader, model, criterion, optimizer)
