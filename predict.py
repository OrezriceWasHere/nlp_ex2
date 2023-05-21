from tqdm import tqdm
from hyper_parameters import *
import torch


def predict(test, model):
    output = []

    for t in test:
        if isinstance(t, list) or isinstance(t, tuple):
            t = [torch.tensor(t_).to(DEVICE).unsqueeze(0) for t_ in t]
            output.append(model(*t).argmax().item())
        else:
            output.append(model(torch.tensor(t).unsqueeze(0).to(DEVICE)).argmax().item())

    return output
