import os.path
import pickle
from argparse import ArgumentParser

import pandas as pd

from utils.constants import DATASET_NAMES
from analysis_scripts.both_experiments_helper import get_models_answers_match_humans
from analysis_scripts.MAZE.plots_helper import clean_IQR
import matplotlib.pyplot as plt
import numpy as np

X_MIN = {"wino": 0.8, "BUG": 0.6}
X_MAX = {"wino": 1.01, "BUG": 0.69}
COLORS = {"anti": "crimson", "pro": "g", "SpanBERT": 'royalblue', "s2e": 'goldenrod'}


def load_MAZE_ids(path_to_results):
    ids = {}
    for ds in DATASET_NAMES:
        path_to_ids = os.path.join(path_to_results, ds, "ids_covered.pkl")
        with open(path_to_ids, 'rb') as f:
            ids[ds] = pickle.load(f)
    return ids


def plot_ds(ds_name, models_result, human_res, out_dir):
    by_relation = human_res.groupby(["relation_to_main"])
    x = np.arange(X_MIN[ds_name], X_MAX[ds_name], 0.01)
    to_add = []
    plt.figure(figsize=(10, 8))

    for model in models_result:
        ds_results = models_result[model][ds_name]
        for category in ["pro", "anti"]:
            to_add.append(round(ds_results[category]/100, 3))
    x = sorted(np.concatenate((x, to_add)))

    times = {}
    ys = {}
    for g in ['anti', 'pro']:
        raw_times = human_res.iloc[by_relation.groups[g]]["time_to_answer"]
        times[g] = clean_IQR(raw_times)
        ys[g] = []

    for fraction in x[:-1]:
        for g in times:
            total = len(times[g])
            frac_i = int(total * fraction)
            ys[g].append(times[g][frac_i])

    for g in ['anti', 'pro']:
        ys[g].append(times[g][-1])
        plt.plot(x, ys[g], label=g, color=COLORS[g])

    for model in models_result:
        for k in ["anti", "pro"]:
            x_val = round(models_result[model][ds_name][k]/100,3)
            index_x = x.index(x_val)
            y_val = ys[k][index_x]
            if k == 'pro':
                m = model
                addition_x = -0.005
                addition_y = -30
            else:
                m = None
                addition_x = -0.03
                addition_y = 15
            plt.plot(x_val, y_val, color=COLORS[model], label=m, marker='o')
            plt.text(x_val + addition_x, y_val + addition_y, f"({x_val},{y_val})")

    plt.legend()
    plt.ylabel("Response Time (in Miliseconds)", fontsize=14, labelpad=8)
    plt.xlabel("% of cases", fontsize=16, labelpad=8)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("MAZE wino results - intersection with models", fontsize=18)
    out_path = os.path.join(out_dir, f"MAZE_{ds_name}_with_models.png")
    plt.savefig(out_path, bbox_inches="tight")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--human_results")
    parser.add_argument("--models_path")
    parser.add_argument("--dataset", choices=DATASET_NAMES)
    parser.add_argument("--out")
    args = parser.parse_args()
    ids_humans = load_MAZE_ids(args.human_results)
    human_res_path = os.path.join(args.human_results, args.dataset, "correct_answers.csv")
    human_answers = pd.read_csv(human_res_path)
    model_answers = get_models_answers_match_humans(args.models_path, ids_humans)
    plot_ds(args.dataset, model_answers, human_answers, args.out)

