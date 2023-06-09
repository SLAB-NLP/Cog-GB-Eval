#!/bin/bash

"""
This script calculate humans results over the QA experiment.
It generates for each participant a table with its scores on the given dataset.
It also generates an aggregated table of the performance of all participants.
The aggregated results will later be used for comparison with models.
To replicate our results, you need to run this script once for each dataset.

Usage:
    analysis_scripts/QA/generate_human_results_table.py --dataset <name> --out_path <path>

Example:
    analysis_scripts/QA/generate_human_results_table.py --dataset wino --out_path experiment_results/processed/humans/QA/wino
    analysis_scripts/QA/generate_human_results_table.py --dataset BUG --out_path experiment_results/processed/humans/QA/BUG
"""

from argparse import ArgumentParser
import os.path
import pandas as pd
import numpy as np
import pickle

from utils.constants import WINO_DATASET_PATH, TABLE_SEPARATOR, \
    WINOGENDER_ORIGINAL_PATH, ANTI_STEREOTYPE_SENTENCES_TYPES, \
    PRO_STEREOTYPE_SENTENCES_TYPES, QA_HUMANS_RAW_RESULTS_DIR, \
    ENROLLMENT_QA_DIR, BUG_ORIGINAL_DATASET_PATH


COLUMNS_FOR_MISTAKES_DF = ["id_sentence", "stereotype", "fraction", "correct_answer",
                           "question", "user_answer", "sentence"]

RESULTS_TABLE_HEADER = ["anti-stereotype", "pro-stereotype", "filler", "ignored", "neutral"]


def parse_table(wino_df, mturk_ids, unique_sentences, qa_answers):
    results = {m_id: {} for m_id in mturk_ids}
    mistakes_df = pd.DataFrame(columns=COLUMNS_FOR_MISTAKES_DF)
    for m_id in mturk_ids:
        contents = qa_answers[qa_answers["mturk_id"] == m_id]
        for _, row in contents.iterrows():
            fraction = row['time_fraction']
            if fraction not in results[m_id]:
                results[m_id][fraction] = {cat: {'num_q': 0, 'num_correct': 0}
                                           for cat in RESULTS_TABLE_HEADER}
            question_category = row['question_category']

            if row['main_entity_gender'] is None:
                row['main_entity_gender'] = wino_df.iloc[row["id_sentence"]]["gender"]
            current_dict, stereotype_row = get_current_dict(row, m_id, question_category,
                                                            fraction, results,
                                                            unique_sentences)

            current_dict['num_q'] += 1
            if row['is_correct']:
                current_dict['num_correct'] += 1
            else:
                if stereotype_row != 'filler':
                    mistake_dict = generate_mistake_dict(fraction, row, stereotype_row)
                    mistakes_df = mistakes_df.append(mistake_dict, ignore_index=True)

    return results, mistakes_df


def generate_mistake_dict(fraction, row, stereotype_row):
    d = {"id_sentence": row["id_sentence"], "stereotype": stereotype_row,
         "fraction": fraction, "sentence": row["sentence"],
         "user_answer": row["user_answer"], "question": row['question'],
         "correct_answer": row["correct_answer"]}
    return d


def get_current_dict(row, m_id, question_category, fraction, results,
                     unique_signal_sentences):
    if question_category == 'filler':
        current_dict = results[m_id][fraction]['filler']
        stereotype_row = 'filler'
    else:
        if args.dataset == 'wino':
            current_dict, stereotype_row = wino_categorization(
                fraction, m_id, question_category, results, row, unique_signal_sentences)
        else:
            current_dict, stereotype_row = BUG_categorization(
                fraction, m_id, question_category, results, row, unique_signal_sentences)

    return current_dict, stereotype_row


def BUG_categorization(fraction, m_id, question_category, results, row,
                       unique_signal_sentences):
    split_cat = question_category.split('_')
    stereotype_row = split_cat[0]
    gender = split_cat[1]

    current_dict = results[m_id][fraction][stereotype_row]
    if stereotype_row in ['anti-stereotype', 'pro-stereotype']:
        unique_signal_sentences[stereotype_row].add(row["id_sentence"])
    return current_dict, stereotype_row


def wino_categorization(fraction, m_id, question_category, results, row,
                        unique_signal_sentences):
    split_cat = question_category.split('.')
    cat = split_cat[1] if split_cat[1] not in {'other', 'main'} else split_cat[2]
    if cat in ANTI_STEREOTYPE_SENTENCES_TYPES:
        stereotype_row = 'anti-stereotype'
    elif cat in PRO_STEREOTYPE_SENTENCES_TYPES:
        stereotype_row = 'pro-stereotype'
    else:
        stereotype_row = 'neutral'
    if stereotype_row != 'neutral' and row['user_answer'] not in {'unknown', 'neutral',
                                                                  row[
                                                                      'main_entity_gender']}:
        stereotype_row = 'ignored'  # ignore non coreference answers
    current_dict = results[m_id][fraction][stereotype_row]
    if stereotype_row in ['anti-stereotype', 'pro-stereotype']:
        if row["sentence"] in winogender_sents_set:
            unique_signal_sentences['winogender'][stereotype_row].add(row["id_sentence"])
        else:
            unique_signal_sentences['winobias'][stereotype_row].add(row["id_sentence"])
    return current_dict, stereotype_row


def analyze(results):
    header = ['ALL'] + RESULTS_TABLE_HEADER + ["anti-stereotype_num", "pro-stereotype_num"]
    fractions = [0.75, 0.5, 0.25]

    f = open(per_participant_results_path, 'wt')
    for mturk_id in results:
        per_user_results(mturk_id, fractions, header, results, f)
    f.write(f"num of participants: {len(results)}")
    f.close()

    aggregate_results(fractions, header, results)


def per_user_results(mturk_id, fractions, header, results, f):
    f.write(f"{mturk_id} performance:\n----------------------------\n")
    df = pd.DataFrame(columns=header, index=fractions)
    for fraction in fractions:
        fraction_row = {}
        frac_dict = results[mturk_id][fraction]
        total_correct = sum(frac_dict[cat]['num_correct'] for cat in RESULTS_TABLE_HEADER)
        total_ans = sum(frac_dict[cat]['num_q'] for cat in RESULTS_TABLE_HEADER)
        update_all_performance(fraction_row, total_ans, total_correct)
        for k in RESULTS_TABLE_HEADER:
            correct = frac_dict[k]['num_correct']
            ans = frac_dict[k]['num_q']
            if k == 'ignored':
                percentage_ignored = ans/(ans+frac_dict['anti-stereotype']['num_q']+frac_dict['pro-stereotype']['num_q'])
                fraction_row[k] = f"{round(percentage_ignored*100,2)}% ({ans})"
            else:
                update_category_performance(fraction_row, k, correct, ans)
        df.loc[fraction] = fraction_row
    write_final_df(df, f)
    f.write('\n\n')


def aggregate_results(fractions, header, results):
    aggregated_table = pd.DataFrame(columns=header, index=fractions)
    file_out = open(aggregated_results_path, 'wt')
    for f in fractions:
        fraction_row = {}
        total_correct = sum(results[mturk_id][f][cat]['num_correct']
                            for cat in RESULTS_TABLE_HEADER for mturk_id in results)
        total_ans = sum(results[mturk_id][f][cat]['num_q']
                        for cat in RESULTS_TABLE_HEADER for mturk_id in results)
        update_all_performance(fraction_row, total_ans, total_correct)
        for k in RESULTS_TABLE_HEADER:
            correct = sum(results[mturk_id][f][k]['num_correct'] for mturk_id in results)
            ans = sum(results[mturk_id][f][k]['num_q'] for mturk_id in results)
            if k == 'ignored':
                percentage_ignored = ans/sum(results[mturk_id][f]['pro-stereotype']['num_q']+results[mturk_id][f]['anti-stereotype']['num_q'] for mturk_id in results)
                fraction_row[k] = f"{round(percentage_ignored*100,2)}% ({ans})"
            else:
                update_category_performance(fraction_row, k, correct, ans)
        aggregated_table.loc[f] = fraction_row
    write_final_df(aggregated_table, file_out)


def write_final_df(results_df, f):
    delta_s = (results_df["pro-stereotype_num"] - results_df["anti-stereotype_num"]).to_numpy()
    percent = np.round([delta_s[i] for i in range(len(delta_s))], 2)
    str_percent = np.array([f"{x}%" for x in percent])
    results_df["delta stereotype (pro minus anti)"] = str_percent
    df = results_df.drop(["anti-stereotype_num", "pro-stereotype_num"], axis=1)
    df = df[['anti-stereotype', 'pro-stereotype', 'delta stereotype (pro minus anti)',
             'ignored', 'filler', 'ALL']]
    df.to_markdown(f)


def update_all_performance(fraction_row, total_ans, total_correct):
    perf = (total_correct / total_ans)*100
    agg_performance_str = f"{round(perf, 2)}% ({total_correct}/{total_ans})"
    fraction_row['ALL'] = agg_performance_str


def update_category_performance(fraction_row, k, num_correct, ans):
    if ans == 0:
        return
    perf = (num_correct / ans)*100
    bias_signal_performance_str = f"{round(perf, 2)}% ({num_correct}/{ans})"
    bias_signal_performance = round(perf, 2)
    fraction_row[k] = bias_signal_performance_str
    if k in ["anti-stereotype", "pro-stereotype"]:
        fraction_row[f"{k}_num"] = bias_signal_performance


def generate_tables(all_df, qa_answers, enrollment):
    ids_to_analyze = sorted(list(set(enrollment["mturk_id"])))
    if args.dataset == 'wino':
        unique_sentences = {d: {c: set() for c in ('pro-stereotype', 'anti-stereotype')}
                            for d in ('winogender', 'winobias')}
    else:
        unique_sentences = {c: set() for c in ('pro-stereotype', 'anti-stereotype')}
    table, mistakes_df = parse_table(all_df, ids_to_analyze, unique_sentences, qa_answers)
    analyze(table)
    dataset_coverage_analyze(unique_sentences)
    mistakes_df.to_csv(mistakes_df_path)


def dataset_coverage_analyze(unique_sentences):
    len_ids = {}
    for d in unique_sentences:
        if type(unique_sentences[d]) == dict:
            len_ids[d] = {}
            for c in unique_sentences[d]:
                len_ids[d][c] = len(unique_sentences[d][c])
        else:
            len_ids[d] = [len(unique_sentences[d])]
    df_numbers = pd.DataFrame(len_ids)
    with open(dataset_coverage_path, 'w') as f:
        df_numbers.to_markdown(f)
    with open(sentences_ids_covered_path, 'wb') as f:
        pickle.dump(unique_sentences, f)


def load_dataset(dataset_name):
    if dataset_name == 'wino':
        data_df = pd.read_csv(WINO_DATASET_PATH, TABLE_SEPARATOR)
    else:
        data_df = pd.read_csv(BUG_ORIGINAL_DATASET_PATH, encoding='latin-1')
    return data_df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["wino", "BUG"],
                        help="name of dataset to analyze")
    parser.add_argument("--out_path", help="Path to save all of the script output")
    args = parser.parse_args()

    # generate output paths
    per_participant_results_path = os.path.join(args.out_path, "per_participant.txt")
    aggregated_results_path = os.path.join(args.out_path, "aggregated.md")
    dataset_coverage_path = os.path.join(args.out_path, "dataset_coverage.md")
    sentences_ids_covered_path = os.path.join(args.out_path, "ids_covered.pkl")
    mistakes_df_path = os.path.join(args.out_path, "mistakes_df.csv")

    # load data
    all_data_df = load_dataset(args.dataset)
    winogender_df = pd.read_csv(WINOGENDER_ORIGINAL_PATH, TABLE_SEPARATOR)
    winogender_sents_set = set(winogender_df["sentence"])
    qa_answers_df = pd.read_csv(os.path.join(QA_HUMANS_RAW_RESULTS_DIR, f"{args.dataset}.csv"))
    enrollment_df = pd.read_csv(os.path.join(ENROLLMENT_QA_DIR, f"{args.dataset}.csv"))

    generate_tables(all_data_df, qa_answers_df, enrollment_df)
