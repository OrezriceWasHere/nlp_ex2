from tqdm import tqdm
from hyper_parameters import *
import torch


def predict(test, model):
    output = []

    for i in range(0, len(test), 64):
        batch = test[i:i + 64]
        output.extend(model(batch).argmax(dim=1).tolist())

    return output
