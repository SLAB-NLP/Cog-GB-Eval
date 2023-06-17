from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from utils.constants import DATASET_NAMES

MARKERS = {'Humans': '.', 'SpanBERT': '*', 's2e': '+'}
LEGEND_MARKER_SIZE = {'Humans': 13, 'SpanBERT': 10, 's2e': 10}
COLORS = {'wino': "mediumblue", 'BUG': "darkorange"}


def load_results(path_to_md):
    df = pd.read_table(path_to_md, sep="|", header=0, index_col=1, skipinitialspace=True)
    df = df.dropna(axis=1, how='all').iloc[1:]
    df.columns = df.columns.str.strip()
    results = {ds: {} for ds in DATASET_NAMES}
    for name, row in df.iterrows():
        name = name.strip()
        if name.startswith("0."):
            name = f"Human {name}"
        results["wino"][name] = {key: float(row[f"wino-{key}"]) for key in ["pro", "anti"]}
        results["BUG"][name] = {key: float(row[f"BUG-{key}"]) for key in ["pro", "anti"]}

    return results


def draw_results(results):
    for ds in DATASET_NAMES:
        for k in results[ds]:
            pro = results[ds][k]["pro"]
            anti = results[ds][k]["anti"]
            # theta = np.degrees(np.arctan(pro / anti))
            if 'Human' in k:
                size = LEGEND_MARKER_SIZE["Humans"]
                marker = MARKERS["Humans"]
                alpha = float(k.split(' ')[1])
                if ds == 'wino':
                    sub = 1 if alpha==0.75 else 2.6
                    plus_anti = 0 if alpha!=0.75 else 1
                    plt.text(anti+plus_anti, pro-sub, f"{alpha}",size=9)
                elif ds == 'BUG':
                    plt.text(anti, pro + 0.7, f"{alpha}", size=9)
            else:
                size = LEGEND_MARKER_SIZE[k]
                marker = MARKERS[k]
            plt.plot(anti, pro, color=COLORS[ds], marker=marker, markersize=size)


def plot_results(all_results, out_path):
    plt.figure()
    draw_results(all_results)
    plt.plot(np.arange(50,105,5), np.arange(50,105,5), color='black', linestyle='--')
    plt.xlabel("anti-stereotype accuracy")
    plt.xticks(np.arange(50,105,5))
    plt.ylabel("pro-stereotype accuracy")
    plt.yticks(np.arange(50,105,5))

    lines_for_legend = [Line2D([0], [0], marker=MARKERS[key], color='black', label=key,
                               linestyle='None', markersize=LEGEND_MARKER_SIZE[key])
                        for key in ['Humans', 's2e', 'SpanBERT']]
    patch_for_legend = [Patch(facecolor='mediumblue', label='Wino'),
                        Patch(facecolor='darkorange', label='BUG')]
    legend_elements = lines_for_legend + patch_for_legend

    plt.legend(handles=legend_elements)
    plt.grid(alpha=0.5)
    plt.savefig(out_path, bbox_inches="tight")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--in_results_md", help="path to markdown file with results")
    parser.add_argument("--out", help="path to output plot to be saved")

    args = parser.parse_args()

    results_dict_for_plot = load_results(args.in_results_md)
    plot_results(results_dict_for_plot, args.out)

