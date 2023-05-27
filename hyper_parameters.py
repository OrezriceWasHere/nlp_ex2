import torch

LR = None  # 1e-4
EMBEDDING_SIZE = 50
NO_WORD = "~U~"
NER_CLASS_TO_INDEX = {'O': 0, 'ORG': 1, 'MISC': 2, 'PER': 3, 'LOC': 4}

POS_CLASS_TO_INDEX = {'CC': 0, 'CD': 1, 'DT': 2, 'EX': 3, 'FW': 4, 'IN': 5, 'JJ': 6, 'JJR': 7, 'JJS': 8, 'LS': 9,
                      'MD': 10, 'NN': 11, 'NNS': 12, 'NNP': 13, 'NNPS': 14, 'PDT': 15, 'POS': 16, 'PRP': 17, 'PRP$': 18,
                      'RB': 19, 'RBR': 20, 'RBS': 21, 'RP': 22, 'SYM': 23, 'TO': 24, 'UH': 25, 'VB': 26, 'VBD': 27,
                      'VBG': 28, 'VBN': 29, 'VBP': 30, 'VBZ': 31, 'WDT': 32, 'WP': 33, 'WP$': 34,
                      'WRB': 35}  # thanks to GPT for creating this dictionary!

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 if DEVICE == torch.device("cuda:0") else 8
PROB_UNQ = 0.07
DROPOUT = 0.3
EPOCHS = 5
CHARACTER_EMBEDDING_SIZE = 30

CHAR_LSTM_HIDDEN_SIZE = 25

LSTM_HIDDEN_SIZE = 50
