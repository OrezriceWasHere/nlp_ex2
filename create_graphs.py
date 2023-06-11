from bilstmTrain import do_it

import matplotlib.pyplot as plt


def save_graph_to_file(points_list, filename):
    fig, ax = plt.subplots()

    for i, points in enumerate(points_list):
        ax.plot(points, label=chr(i + ord('a')))

    ax.legend()

    fig.savefig(f'{filename}.png')


def main():
    for task in ['ner', 'pos']:
        loss, accuracy = [], []
        for option in ['a', 'b', 'c', 'd']:
            l, a = do_it(task, option)[1]
            loss.append(l)
            accuracy.append(a)

        save_graph_to_file(loss, f'{task}_loss')
        save_graph_to_file(accuracy, f'{task}_accuracy')


if __name__ == "__main__":
    main()
