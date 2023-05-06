import os.path
from argparse import ArgumentParser
import pickle
import json
import pandas as pd

from utils.constants import DATASET_NAMES, BUG_ORIGINAL_DATASET_PATH, WINO_DATASET_PATH, \
    TABLE_SEPARATOR, PRO_STEREOTYPE_SENTENCES_TYPES

from analysis_scripts.both_experiments_helper import get_models_answers_match_humans


# def get_models_answers_match_humans(models_path, humans_covered_sentences):
#     """ Retrieves models score only on the sentences humans annotated """
#     BUG_ds = pd.read_csv(BUG_ORIGINAL_DATASET_PATH, encoding='latin-1')
#     wino_ds = pd.read_csv(WINO_DATASET_PATH, sep=TABLE_SEPARATOR)
#     true_val, false_val = 1, -1
#     wino_ds["stereotype"] = wino_ds["type"].apply(
#         lambda t: true_val if t in PRO_STEREOTYPE_SENTENCES_TYPES else false_val)
#     original = {"BUG": BUG_ds, "wino": wino_ds}
#
#     with open(models_path, 'r') as f_in:
#         models_dict = json.load(f_in)
#
#     all_results = {}
#     for k in models_dict:
#         split_k = k.split('-')
#         ds_name, model_name = split_k[0], split_k[1]
#         if model_name not in all_results:
#             all_results[model_name] = {}
#         all_results[model_name][ds_name] = {}
#         model_results = models_dict[k]
#         humans_ids = unite_human_ids(ds_name, humans_covered_sentences)
#         anti_performance, pro_performance = get_pro_anti_performance(
#             ds_name, humans_ids, model_results, original, true_val, false_val)
#         all_results[model_name][ds_name]["pro"] = round(pro_performance * 100, 1)
#         all_results[model_name][ds_name]["anti"] = round(anti_performance * 100, 1)
#         all_results[model_name][ds_name]["delta"] = round((pro_performance - anti_performance) * 100, 1)
#     return all_results
#
#
# def get_pro_anti_performance(ds_name, humans_ids, model_results, original_df,
#                              true_val, false_val):
#     correct_intersect = set(model_results['correct']).intersection(humans_ids)
#     incorrect_intersect = set(model_results['incorrect']).intersection(humans_ids)
#     correct_lines = original_df[ds_name].iloc[list(correct_intersect)]
#     incorrect_lines = original_df[ds_name].iloc[list(incorrect_intersect)]
#     pro_correct = correct_lines[correct_lines["stereotype"] == true_val]
#     pro_incorrect = incorrect_lines[incorrect_lines["stereotype"] == true_val]
#     anti_correct = correct_lines[correct_lines["stereotype"] == false_val]
#     anti_incorrect = incorrect_lines[incorrect_lines["stereotype"] == false_val]
#     pro_performance = len(pro_correct) / (len(pro_incorrect) + len(pro_correct))
#     anti_performance = len(anti_correct) / (len(anti_incorrect) + len(anti_correct))
#     return anti_performance, pro_performance


def unite_human_ids(humans_covered_sentences):
    humans_ids = {}
    for ds_name in humans_covered_sentences:
        humans_ids[ds_name] = set()
        for key in humans_covered_sentences[ds_name]:
            if type(humans_covered_sentences[ds_name][key]) == dict:
                for inner_dict in humans_covered_sentences[ds_name][key]:
                    humans_ids[ds_name].update(humans_covered_sentences[ds_name][key][inner_dict])
            else:
                humans_ids[ds_name].update(humans_covered_sentences[ds_name][key])
    return humans_ids


def generate_combined_tables(models_res, human_res):
    header = ["wino-pro", "wino-anti", "wino-delta", "BUG-pro", "BUG-anti", "BUG-delta"]
    final_df = pd.DataFrame(columns=header)
    combined = human_res
    combined.update(models_res)
    for name in combined:
        line = {f"{dataset}-{cat}": combined[name][dataset][cat]
                for dataset in combined[name]
                for cat in combined[name][dataset]
                }
        final_df.loc[name] = line
    return final_df


def get_human_ids_covered():
    ids = {}
    for ds in DATASET_NAMES:
        humans_coverage_path = os.path.join(args.humans_path, ds, "ids_covered.pkl")
        with open(humans_coverage_path, 'rb') as f:
            ids[ds] = pickle.load(f)
    return ids


def get_human_results(path_to_humans_results):
    paths = {ds: os.path.join(path_to_humans_results, ds, "aggregated.md")
             for ds in DATASET_NAMES}
    human_res_format = {str(fraction): {ds: {} for ds in DATASET_NAMES} for fraction in [0.75, 0.5, 0.25]}
    for ds in paths:
        df = pd.read_table(paths[ds], sep="|", header=0, index_col=1, skipinitialspace=True)
        df = df.dropna(axis=1, how='all').iloc[1:]
        df.columns = df.columns.str.strip()
        for fraction, row in df.iterrows():
            frac = fraction.strip()
            for cat in ["pro", "anti"]:
                human_res_format[frac][ds][cat] = float(row[f"{cat}-stereotype"].split(" ")[0][:-1])
            human_res_format[frac][ds]["delta"] = human_res_format[frac][ds]["pro"] - human_res_format[frac][ds]["anti"]
    return human_res_format


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--models_path", help="path for model results")
    parser.add_argument("--humans_path", help="path for human results dir")
    parser.add_argument("--out", help="path for output")
    args = parser.parse_args()

    human_ids = unite_human_ids(get_human_ids_covered())
    models_results_match = get_models_answers_match_humans(args.models_path, human_ids)
    human_results = get_human_results(args.humans_path)
    final_results_combined = generate_combined_tables(models_results_match, human_results)

    with open(args.out, 'wt') as f:
        final_results_combined.to_markdown(f)

