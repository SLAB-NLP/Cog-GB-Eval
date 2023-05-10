import os.path
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

from utils.constants import US_LABOR_PATH, WINO_STATS_PATH, WINO_DATASET_PATH, \
    BUG_ORIGINAL_DATASET_PATH, TABLE_SEPARATOR, DATASET_NAMES, MODEL_NAMES

FEMALE_PCT_KEY = 'bls_pct_female'


def get_female_percentage(occupation, occupations_wino_stats, us_stats):
    if occupation in set(occupations_wino_stats["occupation"]):
        return occupations_wino_stats[occupations_wino_stats["occupation"] == occupation][FEMALE_PCT_KEY].iloc[0]

    total = 0
    women = 0
    mapping = {'educator': 'education', 'secretary': 'secretaries', 'doctor': 'physicians'}
    if occupation in mapping:
        occupation = mapping[occupation]
    for j, prof_row in us_stats.iterrows():
        if pd.isna(prof_row["Occupation"]):
            continue
        if occupation.lower() in prof_row["Occupation"].lower() and not pd.isna(
                prof_row["Women"]):
            total += prof_row["Total employed"]
            women += (prof_row["Women"] / 100) * prof_row["Total employed"]
    if total == 0:
        female_percent = -999
    else:
        female_percent = round(women / total, 4)*100
    return female_percent


def plot(fraction_groups, human_mistakes_df, occupations_wino_stats, us_stats, out_path):
    ys = []
    yerr = []

    for dataset in human_mistakes_df:
        print(dataset)
        groups = fraction_groups[dataset]
        df = human_mistakes_df[dataset]

        dists_from_half = []
        for frac in groups:
            if frac != 0.5:
                continue
            print('\t', frac)
            ds_frac = df.iloc[groups[frac]]
            non_stereotypes = 0
            for i, row in ds_frac.iterrows():
                profession = row["question"].split(' ')[-1].replace('?', '')
                women_percentage = get_female_percentage(profession, occupations_wino_stats, us_stats)
                if women_percentage == -999:
                    continue
                if (row["user_answer"] == 'female' and women_percentage < 50) \
                        or (row["user_answer"] == 'male' and women_percentage > 50):
                    non_stereotypes += 1
                    continue
                dists_from_half.append(abs(women_percentage - 50))
            ys.append(round(np.average(dists_from_half), 4))
            yerr.append(round(np.std(dists_from_half), 4))
            print('\t', 'avg_dist:', round(np.average(dists_from_half), 4))
            print('\t', 'std_dist:', round(np.std(dists_from_half), 4))
            print('\t', 'non-stereotype mistakes:',
                  round(non_stereotypes / len(ds_frac), 2))

        dists_from_half = []
        for model in MODEL_NAMES:
            print(model)
            key = f"{dataset}-{model}"
            incorrect_models = set(models_res[key]['incorrect'])
            for uid in incorrect_models:
                if dataset == 'BUG':
                    row = BUG_df[BUG_df["uid"] == uid]
                    profession = row["profession"].iloc[0]
                else:
                    row = wino_df[wino_df["Unnamed: 0"] == uid]
                    profession = row["main_entity_occupation"].iloc[0]
                women_percentage = get_female_percentage(profession, occupations_wino_stats, us_stats)
                if women_percentage == -999:
                    continue
                dists_from_half.append(abs(women_percentage - 50))
            ys.append(round(np.average(dists_from_half), 4))
            yerr.append(round(np.std(dists_from_half), 4))

    results_mean = pd.DataFrame(columns=["Wino", "BUG"])
    results_std = pd.DataFrame(columns=["Wino", "BUG"])
    results_mean["BUG"] = pd.Series(ys[:3])
    results_mean["Wino"] = pd.Series(ys[3:])
    x_axis = ["50% human RT", "s2e", "SpanBERT"]
    results_mean.index = pd.Series(x_axis)


    # results_std["category"] = pd.Series(["0.25 human", "0.5 human", "0.75 human"])
    results_std["BUG"] = pd.Series(yerr[:3])
    results_std["Wino"] = pd.Series(yerr[3:])
    results_std.index = pd.Series(x_axis)

    results_mean.plot.bar(yerr=results_std, capsize=4, rot=0)
    plt.ylabel("Stereotype Confidence")
    plt.ylim([0, 50])
    plt.savefig(out_path)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--models_path")
    parser.add_argument("--humans_path")
    parser.add_argument("--out")

    args = parser.parse_args()

    with open(args.models_path, 'r') as f:
        models_res = json.load(f)

    professions_csv_us_stats = pd.read_csv(US_LABOR_PATH, encoding='cp1252')
    wino_df = pd.read_csv(WINO_DATASET_PATH, sep=TABLE_SEPARATOR)
    BUG_df = pd.read_csv(BUG_ORIGINAL_DATASET_PATH, encoding='latin-1')
    wino_stats = pd.read_csv(WINO_STATS_PATH, sep=TABLE_SEPARATOR)

    human_mistakes = {}
    for ds in DATASET_NAMES:
        path = os.path.join(args.humans_path, ds, "mistakes_df.csv")
        human_mistakes[ds] = pd.read_csv(path)

    groups_by_fraction = {key: human_mistakes[key].groupby('fraction').groups
                          for key in human_mistakes}

    out_path = os.path.join(args.out, "stereotype_confidence.png")
    plot(groups_by_fraction, human_mistakes, wino_stats, professions_csv_us_stats, out_path)

