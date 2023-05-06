#!/bin/bash

"""
This script analyze models results over datasets.
It creates for each model and each dataset, a list of incorrect predictions sentences ids,
and a list of correct predictions sentences ids.
This script parse a jsonl file of model results as it appears in MODELS_RAW_RESULTS_DIR.
If you want to parse your own model results, you need to add them to
MODELS_RAW_RESULTS_DIR in the same format as the existing model results (s2e and SpanBERT).

Usage:
    analyze_models/generate_result_ids_lists.py --out_path <path> --model_name s2e --model_name SpanBERT
"""

import os
from tqdm import tqdm
import jsonlines
import json
import pandas as pd
from itertools import product
from argparse import ArgumentParser
from utils.BUG.map_sentences import get_closest_sentence
from utils.constants import MODELS_RAW_RESULTS_DIR, WINO_DATASET_PATH, \
    TABLE_SEPARATOR, BUG_ORIGINAL_DATASET_PATH, PRO_STEREOTYPE_SENTENCES_TYPES, \
    DATASET_NAMES, MODEL_NAMES


def eval_single_model(dataset_name, model, data_df):
    """
    generate 2 lists of sentences uid. One holds the uid of sentences that models predict
    correctly, and the other list for ids of incorrect model prediction.
    :param dataset_name: The name of the dataset to analyze.
    :param model: The name of the models to analyze.
    :param data_df: Panda DataFrame of the dataset to analyze.
    :return: list of correct and list of incorrect ids.
    """
    path_to_res = os.path.join(MODELS_RAW_RESULTS_DIR, model, f"{dataset_name}.jsonl")

    with jsonlines.open(path_to_res) as f:
        correct_lst = []
        incorrect_lst = []
        for line in tqdm(f.iter()):
            sentence, tokens = reconstruct_sentence_from_tokens(line)
            clusters = line["clusters"]

            if dataset_name == 'BUG':
                uid = get_closest_sentence(data_df, sentence)
                original_row = data_df[data_df["uid"] == uid].iloc[0]
                main_entity = original_row["profession"]
                pronoun = original_row["g"]
            else:
                sentence = clear_wino_sentence(sentence, data_df)
                original_row = data_df[data_df["sentence"] == sentence].iloc[0]
                main_entity = original_row["main_entity_occupation"]
                pronoun = original_row["pronoun"]
                uid = original_row.name
                assert original_row.name == original_row["Unnamed: 0"]

            if len(clusters) == 0:
                incorrect_lst.append(int(uid))
                continue
            is_correct = is_overlapping_clusters(clusters, main_entity, pronoun, tokens)
            if is_correct:
                correct_lst.append(int(uid))
            else:
                incorrect_lst.append(int(uid))

        return correct_lst, incorrect_lst


def is_overlapping_clusters(clusters, main_entity, pronoun, tokens):
    """ Check for overlapping clusters, for either correct or incorrect prediction """
    for cluster in clusters:
        c_pronoun, c_entity = False, False
        for node in cluster:
            words_in_cluster = ' '.join(tokens[node[0]:node[1] + 1])
            if main_entity in words_in_cluster:
                c_entity = True
            if pronoun in words_in_cluster:
                c_pronoun = True
        if c_pronoun and c_entity:
            return True
    return False


def clear_wino_sentence(sentence, original_dataset):
    """ Some cleaning for the wino dataset sentences """
    sentence = sentence.replace(' .', '.').replace('can not', 'cannot'). \
        replace(' ,', ',').replace(' n\'t', 'n\'t').replace(' - ', '-'). \
        replace(', 0', ',0').replace(' ;', ';').replace('don\'thing', 'do nothing')
    if sentence not in set(original_dataset["sentence"]):
        sentence = sentence.replace(' \'s', '\'s')
    return sentence


def reconstruct_sentence_from_tokens(line):
    """ reconstruct the sentence from the model prediction tokens """

    if "tokens" in line:
        tokens = line["tokens"]
        if "sentence" in line:
            sentence = line["sentence"]
        else:
            sentence = line["sentence_text"]
    else:
        tokens = line["document"]
        sentence = ' '.join(tokens)
    return sentence, tokens


def analyze_models(out_path, model_names):
    """
    For each model and each dataset, analyze correct and incorrect predictions.
    Generate a json file containing a list of incorrect and a list of correct predictions,
    for each model and each dataset.

    :param out_path: path to dump the generated results.
    :param model_names: names of the models to analyze.
    """

    all_results = {}
    for dataset, model in product(DATASET_NAMES, model_names):
        if dataset == 'BUG':
            original_dataset = pd.read_csv(BUG_ORIGINAL_DATASET_PATH, encoding='latin-1')
        else:
            original_dataset = pd.read_csv(WINO_DATASET_PATH, TABLE_SEPARATOR)
            original_dataset["stereotype"] = original_dataset["type"].apply(lambda t: t in PRO_STEREOTYPE_SENTENCES_TYPES)
        correct_lst, incorrect_lst = eval_single_model(dataset, model, original_dataset)
        all_results[f"{dataset}-{model}"] = {"correct": correct_lst, "incorrect": incorrect_lst}
    with open(out_path, 'wt') as f:
        json.dump(all_results, f)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--out_path", help="path to save model results")
    parser.add_argument("--model_name", default=MODEL_NAMES, action='append',
                        help="name of models to evaluate on")

    args = parser.parse_args()
    analyze_models(args.out_path, args.model_name)
