import torch

LR = None  # 1e-4
EMBEDDING_SIZE = 500
START_PAD = "~s~"
END_PAD = "~e~"
NO_WORD = "~u~"
WINDOW = 5
NER_CLASS_TO_INDEX = {'O': 0, 'ORG': 1, 'MISC': 2, 'PER': 3, 'LOC': 4}
NER_LAYERS = [EMBEDDING_SIZE * WINDOW, 500, len(NER_CLASS_TO_INDEX)]
POS_CLASS_TO_INDEX = {'CC': 0, 'CD': 1, 'DT': 2, 'EX': 3, 'FW': 4, 'IN': 5, 'JJ': 6, 'JJR': 7, 'JJS': 8, 'LS': 9,
                      'MD': 10, 'NN': 11, 'NNS': 12, 'NNP': 13, 'NNPS': 14, 'PDT': 15, 'POS': 16, 'PRP': 17, 'PRP$': 18,
                      'RB': 19, 'RBR': 20, 'RBS': 21, 'RP': 22, 'SYM': 23, 'TO': 24, 'UH': 25, 'VB': 26, 'VBD': 27,
                      'VBG': 28, 'VBN': 29, 'VBP': 30, 'VBZ': 31, 'WDT': 32, 'WP': 33, 'WP$': 34,
                      'WRB': 35}  # thanks to GPT for creating this dictionary!
POS_LAYERS = [EMBEDDING_SIZE * WINDOW, 500, len(POS_CLASS_TO_INDEX)]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 if DEVICE == torch.device("cuda:0") else 8
PROB_UNQ = 0.07
DROPOUT = 0.3
AUGMENTATION_COUNT = 3
EPOCHS = 0#150 // AUGMENTATION_COUNT
CHARACTER_EMBEDDING_SIZE = 30
MAX_CHARACTERS = 15  # helps as padding for the CNN too!
FILTERS_NUMBER = 30
FILTER_SIZE = 3

CHAR_NER_LAYERS = [EMBEDDING_SIZE * WINDOW + WINDOW * FILTERS_NUMBER, 500,
                   len(NER_CLASS_TO_INDEX)]
