import os.path
from argparse import ArgumentParser
from utils.constants import DATASET_NAMES
import numpy as np
import matplotlib.pyplot as plt

DELTAS_COLORS = {"wino": "brown", "BUG": "gold"}


def pyplot_config():
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title("Delta-stereotype (<anti - pro>)", fontsize=20, pad=13)
    plt.ylabel("Difference (in Miliseconds)", fontsize=14, labelpad=8)
    plt.xlabel("% of cases", fontsize=16, labelpad=8)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend()


def plot_deltas(path_to_results):
    deltas_dict = {}
    x = np.arange(0.5, 1.01, 0.01)
    plt.figure(figsize=(10, 8))

    for ds in DATASET_NAMES:
        deltas_path = os.path.join(path_to_results, ds, 'deltas.npy')
        deltas_dict[ds] = np.load(deltas_path)
        plt.plot(x, deltas_dict[ds], color=DELTAS_COLORS[ds], label=ds)
    pyplot_config()
    path = os.path.join(path_to_results, "delta_plot.png")
    plt.savefig(path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--results_dir",
                        help="where deltas are saved and output plots will be saved.")
    args = parser.parse_args()

    plot_deltas(args.results_dir)
