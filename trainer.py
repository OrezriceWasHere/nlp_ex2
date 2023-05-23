from tqdm import tqdm
from hyper_parameters import *
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def train(train_loader, test_loader, model, criterion, optimizer, name, epochs, ignore_first):
    # Prepare training

    loss_per_epoch = []
    accuracy_per_epoch = []

    # Start training
    for epoch in range(epochs):
        predictions = []
        truths = []
        total_loss_train = 0

        print("Epoch: ", epoch)
        print("\n----------------")
        print("Train:")
        model.train()
        for text, label in (pbar := tqdm(train_loader)):
            pbar.set_description(f"Training epoch {epoch}")

            #input(":(")
            output = model(*text)
            #input()
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss_train += batch_loss
            predictions.extend(output.argmax(dim=1).tolist())
            truths.extend(label.tolist())
        print(f'Train Loss: {total_loss_train / len(train_loader): .3f}')
        print(
            f'Train acc: {len([0 for t, p in zip(truths, predictions) if t == p]) / len(truths) if not ignore_first else len([0 for t, p in zip(truths, predictions) if t == p and t != 0]) / len([0 for t in truths if t != 0])}')
        # print(classification_report(truths, predictions, target_names=list(NER_CLASS_TO_INDEX.keys())))

        predictions = []
        truths = []
        total_loss_test = 0

        print("\n----------------")
        print("Test: ")

        with torch.no_grad():
            model.eval()
            for text, label in (pbar := tqdm(test_loader)):
                pbar.set_description(f"Evaluation epoch {epoch}")
                output = model(*text)
                loss = criterion(output, label)
                batch_loss = loss.item()
                total_loss_test += batch_loss
                predictions.extend(output.argmax(dim=1).tolist())
                truths.extend(label.tolist())

        print(f'Test Loss: {total_loss_test / len(test_loader): .3f}')
        loss_per_epoch.append(total_loss_test / len(test_loader))
        accuracy_per_epoch.append(
            len([0 for t, p in zip(truths, predictions) if t == p]) / len(truths) if not ignore_first else
            len([0 for t, p in zip(truths, predictions) if t == p and t != 0]) / len([0 for t in truths if t != 0]))
        print(f'Test accuracy: {accuracy_per_epoch[-1]}')

        # print(classification_report(truths, predictions, target_names=list(NER_CLASS_TO_INDEX.keys())))

        # if epoch % 20 == 0:
        #     matrix = confusion_matrix(truths, predictions)
        #     cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(NER_CLASS_TO_INDEX.keys()))
        #     cm_display.plot()
        #     plt.show()

    plt.plot(loss_per_epoch)
    plt.savefig(name + '_loss.png')

    plt.clf()

    plt.plot(accuracy_per_epoch)
    plt.savefig(name + '_accuracy.png')
