import pandas as pd
import json

from utils.constants import BUG_ORIGINAL_DATASET_PATH, WINO_DATASET_PATH, \
    TABLE_SEPARATOR, PRO_STEREOTYPE_SENTENCES_TYPES


def get_models_answers_match_humans(models_path, humans_ids):
    """ Retrieves models score only on the sentences humans annotated """
    BUG_ds = pd.read_csv(BUG_ORIGINAL_DATASET_PATH, encoding='latin-1')
    wino_ds = pd.read_csv(WINO_DATASET_PATH, sep=TABLE_SEPARATOR)
    true_val, false_val = 1, -1
    wino_ds["stereotype"] = wino_ds["type"].apply(
        lambda t: true_val if t in PRO_STEREOTYPE_SENTENCES_TYPES else false_val)
    original = {"BUG": BUG_ds, "wino": wino_ds}

    with open(models_path, 'r') as f_in:
        models_dict = json.load(f_in)

    all_results = {}
    for k in models_dict:
        split_k = k.split('-')
        ds_name, model_name = split_k[0], split_k[1]
        if model_name not in all_results:
            all_results[model_name] = {}
        all_results[model_name][ds_name] = {}
        model_results = models_dict[k]
        anti_performance, pro_performance = get_pro_anti_performance(
            ds_name, humans_ids[ds_name], model_results, original, true_val, false_val)
        all_results[model_name][ds_name]["pro"] = round(pro_performance * 100, 1)
        all_results[model_name][ds_name]["anti"] = round(anti_performance * 100, 1)
        all_results[model_name][ds_name]["delta"] = round((pro_performance - anti_performance) * 100, 1)
    return all_results


def get_pro_anti_performance(ds_name, humans_ids, model_results, original_df,
                             true_val, false_val):
    correct_intersect = set(model_results['correct']).intersection(humans_ids)
    incorrect_intersect = set(model_results['incorrect']).intersection(humans_ids)
    correct_lines = original_df[ds_name].iloc[list(correct_intersect)]
    incorrect_lines = original_df[ds_name].iloc[list(incorrect_intersect)]
    pro_correct = correct_lines[correct_lines["stereotype"] == true_val]
    pro_incorrect = incorrect_lines[incorrect_lines["stereotype"] == true_val]
    anti_correct = correct_lines[correct_lines["stereotype"] == false_val]
    anti_incorrect = incorrect_lines[incorrect_lines["stereotype"] == false_val]
    pro_performance = len(pro_correct) / (len(pro_incorrect) + len(pro_correct))
    anti_performance = len(anti_correct) / (len(anti_incorrect) + len(anti_correct))
    return anti_performance, pro_performance
