import torch

LR = 1e-4
EMBEDDING_SIZE = 50
START_PAD = "<s>"
END_PAD = "</s>"
NO_WORD = "<unk>"
WINDOW = 5
NER_CLASS_TO_INDEX = {'O': 0, 'ORG': 1, 'MISC': 2, 'PER': 3, 'LOC': 4}
NER_LAYERS = [EMBEDDING_SIZE * WINDOW, 500, len(NER_CLASS_TO_INDEX)]
EPOCHS = 500
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100 if DEVICE == torch.device("cuda:0") else 5
PROB_UNQ = 0.07
DROPOUT = 0.3
