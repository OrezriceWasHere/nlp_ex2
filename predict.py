from tqdm import tqdm
from hyper_parameters import *
import torch


def predict(test, model):
    output = []

    for a, b, c, d in test:
        d = [torch.tensor(d_).to(DEVICE) for d_ in d]
        output.extend(model(torch.tensor(a).to(DEVICE), torch.tensor(b).to(DEVICE),
                            torch.tensor(c).to(DEVICE), d).argmax(dim=1).tolist())

    return output
