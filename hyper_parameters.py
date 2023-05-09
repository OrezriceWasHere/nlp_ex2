import torch

LR = 1e-4
EMBEDDING_SIZE = 50
START_PAD = "<s>"
END_PAD = "</s>"
NO_WORD = "<unk>"
NO_WORD_EMBEDDING = [0] * 50
WINDOW = 5
NER_CLASS_TO_INDEX = {'O': 0, 'ORG': 1, 'MISC': 2, 'PER': 3, 'LOC': 4}
NER_LAYERS = [EMBEDDING_SIZE * WINDOW, len(NER_CLASS_TO_INDEX)]
EPOCHS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100 if DEVICE == torch.device("cuda:0") else 5
