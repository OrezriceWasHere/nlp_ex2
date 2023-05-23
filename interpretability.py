from hyper_parameters import *
import torch


def explain(network, tuples, char_to_index):
    results = {}
    for i in range(FILTERS_NUMBER):
        results[i] = {}

    for t in tuples:
        indices = [char_to_index[START_PAD]] * 2 + [char_to_index.get(t_) or char_to_index[NO_WORD] for t_ in t] + [
            char_to_index[END_PAD]] * 2
        e = network.embed_characters(torch.tensor(indices).unsqueeze(0).unsqueeze(0).to(DEVICE)).squeeze(0)
        for i, r in enumerate(e):
            results[i][t] = r.item()

    for i in range(FILTERS_NUMBER):
        sorted_dict = sorted(results[i], key=results[i].get, reverse=True)
        highest = sorted_dict[:5]
        lowest = sorted_dict[-5:]
        print(f'{i}: {highest} {lowest}')
