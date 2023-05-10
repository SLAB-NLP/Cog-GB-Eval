import pickle

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


COLORS = {"anti": "crimson", "pro": "g", "neutral": "mediumblue",
          "pronoun-neutral": "chocolate"}

HEADER = ["time_received", "participant_IP", "controller", "item_num", "element_num",
          "type", "group", "word_num", "word", "alternative", "word_side", "correct",
          "reading_time_to_first_answer", "sentence", "total_time_to_correct"]


def calculate_total_success_rate(processed_results_df, out_dir):
    out_path = os.path.join(out_dir, "success_rate.md")
    by_relation_cat_main = processed_results_df.groupby(["relation_to_main"])
    rate = []
    for g in by_relation_cat_main.groups:
        item_df = processed_results_df.iloc[by_relation_cat_main.groups[g]]
        s = item_df['correct'].sum()
        d = {'category': g, 'num_correct': s, 'total': len(item_df),
             'success_rate': s / len(item_df)}
        rate.append(d)
    total_success_df = pd.DataFrame(rate)
    with open(out_path, 'wt') as f:
        total_success_df.to_markdown(f)


def remove_incorrect_lines(df):
    indexes = df[df['correct'] == 0].index
    only_correct_answered = df.drop(indexes)
    only_correct_answered = only_correct_answered.reset_index(drop=True)
    return only_correct_answered


def plot_cdf(df, out_dir, ds_name):
    by_relation = df.groupby(["relation_to_main"])
    x = np.arange(0.5, 1.01, 0.01)
    times = {}
    ys = {}
    for g in by_relation.groups:
        raw_times = df.iloc[by_relation.groups[g]]["time_to_answer"]
        times[g] = clean_IQR(raw_times)
        ys[g] = []

    for fraction in x[:-1]:
        for g in times:
            total = len(times[g])
            frac_i = int(total * fraction)
            ys[g].append(times[g][frac_i])

    pyplot_design(ds_name)
    for g in by_relation.groups:
        ys[g].append(times[g][-1])
        plt.plot(x, ys[g], label=g, color=COLORS[g])
    plt.legend(fontsize=14)

    out_path = os.path.join(out_dir, "cdf.png")
    plt.savefig(out_path, bbox_inches="tight")
    deltas = np.array(ys['anti']) - np.array(ys['pro'])
    deltas_out_path = os.path.join(out_dir, "deltas.npy")
    np.save(deltas_out_path, deltas)


def pyplot_design(ds_name):
    plt.figure(figsize=(6, 4))
    plt.ylabel("Response Time (in Miliseconds)", fontsize=14, labelpad=8)
    plt.xlabel("% of cases", fontsize=16, labelpad=8)
    plt.title(f"{ds_name} MAZE cdf", fontsize=20, pad=13)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)


def clean_IQR(raw_times):
    time = sorted(raw_times)
    q75, q25 = np.percentile(time, [75, 25])
    intr_qr = q75 - q25
    max_g = q75 + (1.5 * intr_qr)
    min_g = q25 - (1.5 * intr_qr)
    idx_max = (np.abs(time - max_g)).argmin()
    idx_min = (np.abs(time - min_g)).argmin()
    clear_time = time[idx_min:idx_max]
    return clear_time


def save_ids_of_analyze(ids_list, out_dir):
    path = os.path.join(out_dir, "ids_covered.pkl")
    with open(path, 'wb') as f:
        pickle.dump(set(ids_list), f)


def save_df(df, out_dir):
    path = os.path.join(out_dir, "correct_answers.csv")
    df.to_csv(path)
