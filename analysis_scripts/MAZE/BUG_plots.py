import os.path
import pandas as pd
from argparse import ArgumentParser
from utils.constants import BUG_ORIGINAL_DATASET_PATH, MAZE_HUMANS_RAW_RESULTS_DIR, \
    BUG_IDS_MAPPING_PATH
import re
import json

from analysis_scripts.MAZE.plots_helper import calculate_total_success_rate, \
    remove_incorrect_lines, plot_cdf, HEADER, save_ids_of_analyze

BUG_STEREOTYPE_MAPPING = {1: "pro", 0: "neutral", -1: "anti"}


def clear_word(word):
    if pd.isna(word):
        return word
    replaced = word.replace(',', '').replace('.', '').replace('!', '').replace(')', '').\
        replace('`', '').replace('#', '').replace(';', '').replace(' - ', '-').\
        replace(':', '').replace('"', '').replace('(', '')
    clear = re.sub(' +', ' ', replaced).strip()
    if len(clear) > 0:
        if clear[-1] == '\'':
            clear = clear[:-1]
        elif clear.endswith('\'s'):
            clear = clear[:-2]
    return clear


def get_special_main_entity(main_entity, sentence):
    special_professions = ['singer-songwriter', 'super-scientist', 'master-physician',
                           'writer-director', 'boat-builder', 'singer-businessman',
                           'actress/singer', 'test-pilot', 'general-in-chief',
                           'stage-manager', 'singer/songwriter']
    for profession in special_professions:
        if profession in sentence.lower():
            main_entity = profession
            break
    return main_entity


def get_mapping():
    with open(BUG_IDS_MAPPING_PATH, 'rt') as f:
        mapping = json.load(f)
    return mapping


def load_data(path_to_csv):
    df = pd.read_csv(path_to_csv, comment='#')
    df.columns = HEADER
    df = df.dropna(subset=['word'])
    df["word"] = df["word"].apply(lambda x: clear_word(x).replace('%2C', '').lower())
    return df


def parse_results(raw_results_df, original_data):
    groups = raw_results_df.groupby(["item_num", "participant_IP"]).groups
    lines = []
    mapping = get_mapping()
    for g in groups:
        item_df = raw_results_df.loc[groups[g]]
        if item_df["type"].iloc[0] == "practice":
            continue
        sentence = item_df.iloc[0]["sentence"].replace('%2C', ',')
        if "artsy gigolo-writer" in sentence or 'his/her' in sentence \
                or 'The patient experienced no grade' in sentence:
            continue
        original_row = original_data[original_data["uid"] == mapping[sentence]].iloc[0]
        d = generate_dictionary(item_df, original_row, sentence)
        lines.append(d)
    results_df = pd.DataFrame(lines)
    return results_df


def generate_dictionary(item_df, original_row, sentence):
    pronoun = original_row["g"]
    main_entity = get_special_main_entity(original_row["profession"], sentence)
    pronoun_index = min(item_df[item_df["word"] == pronoun.lower()]["word_num"])
    results_row = item_df.iloc[pronoun_index]
    main_entity_index = min(
        item_df[item_df["word"] == main_entity.lower()]["word_num"])
    d = {"pronoun": pronoun,
         "gender": original_row["g"],
         "main_entity": main_entity,
         "relation_to_main": BUG_STEREOTYPE_MAPPING[original_row["stereotype"]],
         "dist_to_main": pronoun_index - main_entity_index,
         "time_to_answer": results_row["reading_time_to_first_answer"],
         "correct": results_row["reading_time_to_first_answer"] == results_row[
             "total_time_to_correct"],
         "user_ip": results_row["participant_IP"],
         "sentence_id": original_row['uid'],
         "sentence": sentence,
         "word_side": results_row["word_side"]}
    return d


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--out_path", help="path to save results")
    args = parser.parse_args()
    original_BUG = pd.read_csv(BUG_ORIGINAL_DATASET_PATH, encoding='latin-1')
    raw_df = load_data(os.path.join(MAZE_HUMANS_RAW_RESULTS_DIR, "BUG"))
    processed_df = parse_results(raw_df, original_BUG)
    calculate_total_success_rate(processed_df, args.out_path)
    only_correct = remove_incorrect_lines(processed_df)
    save_ids_of_analyze(only_correct["sentence_id"].values, args.out_path)
    plot_cdf(only_correct, args.out_path, "BUG")
