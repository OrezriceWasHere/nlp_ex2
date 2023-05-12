from tqdm import tqdm
from hyper_parameters import *
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

def train(train_loader, test_loader, model, criterion, optimizer):
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

            output = model(text)
            loss = criterion(output, label)
            optimizer.zero_grad()
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
        if epoch % 20 == 0:
            matrix = confusion_matrix(truths, predictions)
            cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(NER_CLASS_TO_INDEX.keys()))
            cm_display.plot()
            plt.show()
