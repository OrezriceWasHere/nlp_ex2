from tqdm import tqdm
from hyper_parameters import *
import torch


def predict(test, model, with_chars):
    output = []

    for t in test:
        if with_chars:
            t = [torch.tensor(t_).unsqueeze(0).to(DEVICE) for t_ in t]
            output.append(model(*t).argmax().item())
        else:
            output.append(model(torch.tensor(t).unsqueeze(0).to(DEVICE)).argmax().item())

    return output
