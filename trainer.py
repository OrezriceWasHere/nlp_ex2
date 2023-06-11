from tqdm import tqdm
from hyper_parameters import *
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import torch


def calculate_accuracy(truths, predictions, ignore_first):
    return len([0 for t, p in zip(truths, predictions) if t == p]) / len(
        truths) if not ignore_first else len([0 for t, p in zip(truths, predictions) if t == p and t != 0]) / len(
        [0 for t in truths if t != 0])


def eval(test_loader, model, criterion, ignore_first):
    predictions = []
    truths = []
    total_loss_test = 0

    with torch.no_grad():
        model.eval()
        for text, label in test_loader:
            output = model(*text)
            loss = criterion(output, label)
            batch_loss = loss.item()
            total_loss_test += batch_loss
            predictions.extend(output.argmax(dim=1).tolist())
            truths.extend(label.tolist())

    model.train()

    loss, accuracy = total_loss_test / len(test_loader), calculate_accuracy(truths, predictions, ignore_first)

    return loss, accuracy


def train(train_loader, test_loader, model, criterion, optimizer, epochs, ignore_first):
    # Prepare training

    loss_per = []
    accuracy_per = []

    counter = 0

    model.train()

    # Start training
    for epoch in range(epochs):
        predictions = []
        truths = []
        total_loss_train = 0

        print("Epoch: ", epoch)
        print("\n----------------")
        print("Train:")
        for text, label in (pbar := tqdm(train_loader)):
            pbar.set_description(f"Training epoch {epoch}")

            output = model(*text)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss_train += batch_loss
            predictions.extend(output.argmax(dim=1).tolist())
            truths.extend(label.tolist())

            if counter % 500 == 0:
                loss, accuracy = eval(test_loader, model, criterion, ignore_first)

                loss_per.append(loss)
                accuracy_per.append(accuracy)

            counter += 1

        print(f'Train Loss: {total_loss_train / len(train_loader): .3f}')
        print(f'Train acc: {calculate_accuracy(truths, predictions, ignore_first)}')

        print(f'Test accuracy: {eval(test_loader, model, criterion, ignore_first)[1]}')

        print("\n----------------")

        # criterion = torch.nn.CrossEntropyLoss()

    return loss_per, accuracy_per
