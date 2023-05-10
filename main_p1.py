import torch
from tqdm import tqdm
from hyper_parameters import *
from data.word_tagger_dataset import WordTaggerDataset
from sequence_tagger_network import SequenceTaggerNetwork
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

print(f'running on {DEVICE}')


def train(train_loader, test_loader, model, criterion):
    # Prepare training

    # Start training
    for epoch in range(EPOCHS):
        predictions = []
        truths = []
        total_loss_train = 0

        print("Epoch: ", epoch)
        for text, label in (pbar := tqdm(train_loader)):
            pbar.set_description(f"Training epoch {epoch}")
            model.train()
            text = text.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss_train += batch_loss
            predictions.extend(output.argmax(dim=1).tolist())
            truths.extend(label.tolist())
        print("\n----------------")
        print("Train results: ")
        print(f'\nEpochs: {epoch + 1}')
        print(f'Train Loss: {total_loss_train / len(train_loader): .3f}')
        print(classification_report(truths, predictions, target_names=list(NER_CLASS_TO_INDEX.keys())))

        predictions = []
        truths = []
        total_loss_test = 0

        for text, label in (pbar := tqdm(test_loader)):
            pbar.set_description(f"Evaluation epoch {epoch}")
            with torch.no_grad():
                model.eval()
                text = text.to(DEVICE)
                label = label.to(DEVICE)
                output = model(text)
                loss = criterion(output, label)
                batch_loss = loss.item()
                total_loss_test += batch_loss
                predictions.extend(output.argmax(dim=1).tolist())
                truths.extend(label.tolist())
        print("\n----------------")
        print("test results: ")
        print(f'Test Loss: {total_loss_test / len(test_loader): .3f}')
        print(classification_report(truths, predictions, target_names=list(NER_CLASS_TO_INDEX.keys())))
        matrix = confusion_matrix(truths, predictions)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(NER_CLASS_TO_INDEX.keys()))
        cm_display.plot()
        plt.show()

if __name__ == "__main__":
    ner_train_file_path = "data/ner/train"

    # Load word embeddings
    word_to_embedding_dict = {}
    with open("data/embedding/vocab.txt") as vocab_file, open("data/embedding/wordVectors.txt") as vector_file:
        for word, embedding in zip(vocab_file, vector_file):
            word_to_embedding_dict[word.rstrip()] = [float(x) for x in embedding.rstrip().split(" ")]

    # Preprare dataset
    train_dataset = WordTaggerDataset(ner_train_file_path, word_to_embedding_dict, NER_CLASS_TO_INDEX)
    test_dataset = WordTaggerDataset(ner_train_file_path, word_to_embedding_dict, NER_CLASS_TO_INDEX)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Prepare model
    model = SequenceTaggerNetwork(NER_LAYERS)
    model = model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    train(train_loader, test_loader, model, criterion)
