#!/bin/bash

"""
This scripts generates a combined table for humans and models results, comparing human
scores over the QA experiment, and model scores over the exact same sentences humans
covered in the QA experiment.

This script assumes you've already run the generate_human_results_table.py script.

Usage:
    analysis_scripts/QA/generate_combined_results.py --models_path <path_to_models_results_dictionary> --humans_path <path_to_humans_scoring_table> --out <out_path>

Example:
    analysis_scripts/QA/generate_combined_results.py --models_path experiment_results/processed/models/eval.json --humans_path experiment_results/processed/humans/QA/ --out experiment_results/processed/final/QA_combined_table.md

"""

import os.path
from argparse import ArgumentParser
import pickle
import pandas as pd

from utils.constants import DATASET_NAMES
from analysis_scripts.both_experiments_helper import get_models_answers_match_humans


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

