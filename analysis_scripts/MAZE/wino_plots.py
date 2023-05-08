
import os.path
import pandas as pd
from argparse import ArgumentParser
from utils.constants import WINO_DATASET_PATH, MAZE_HUMANS_RAW_RESULTS_DIR, \
    TABLE_SEPARATOR

from analysis_scripts.MAZE.plots_helper import calculate_total_success_rate, \
    remove_incorrect_lines, plot_cdf, HEADER, save_ids_of_analyze, save_df


def load_data(path_to_csv):
    df = pd.read_csv(path_to_csv, comment='#')
    df.columns = HEADER
    df["word"] = df["word"].apply(lambda x: x.replace(".", "").replace(",", ""))
    return df


def parse_results(raw_results_df, original_data):
    groups = raw_results_df.groupby(["item_num", "participant_IP"]).groups
    lines = []
    for g in groups:
        item_df = raw_results_df.iloc[groups[g]]
        sentence = item_df.iloc[0]["sentence"].replace('%2C', ',')
        original_row = original_data[original_data["sentence"] == sentence].iloc[0]
        results_pronoun_row = item_df[item_df["word"] == original_row["pronoun"]].iloc[0]
        if results_pronoun_row["type"] == "practice":
            continue
        d = generate_dictionary(original_row, results_pronoun_row)
        lines.append(d)
    processed_results_df = pd.DataFrame(lines)
    return processed_results_df


def generate_dictionary(original_row, results_row):
    pronoun_index = results_row["word_num"]
    main_entity_index = original_row["main_entity_index"]
    other_entity_index = original_row["other_entity_index"]
    d = {"pronoun": original_row["pronoun"],
         "gender": original_row["gender"],
         "main_entity": original_row["main_entity_occupation"],
         "other_entity": original_row["other_entity_occupation"],
         "relation_to_main": results_row["type"].split('-')[0],
         "relation_to_other": results_row["type"].split('-')[1],
         "dist_to_main": pronoun_index - main_entity_index,
         "dist_to_other": pronoun_index - other_entity_index,
         "time_to_answer": results_row["reading_time_to_first_answer"],
         "correct": results_row["reading_time_to_first_answer"] == results_row[
             "total_time_to_correct"],
         "user_ip": results_row["participant_IP"],
         "sentence_id": original_row['Unnamed: 0'],
         "sentence": original_row["sentence"],
         "word_side": results_row["word_side"]}
    d["closer_entity"] = "main" if d["dist_to_main"] < d["dist_to_other"] else "other"
    d["closer_distance"] = d[f"dist_to_{d['closer_entity']}"]
    if d["relation_to_main"] == 'pronoun':
        d["relation_to_main"] = 'pronoun-neutral'
        d["relation_to_other"] = 'pronoun-neutral'
    d["closer_relation"] = d[f"relation_to_{d['closer_entity']}"]
    return d


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--out_path", help="path to save results")
    args = parser.parse_args()
    original_wino = pd.read_csv(WINO_DATASET_PATH, sep=TABLE_SEPARATOR)
    raw_df = load_data(os.path.join(MAZE_HUMANS_RAW_RESULTS_DIR, "wino.raw"))
    processed_df = parse_results(raw_df, original_wino)
    calculate_total_success_rate(processed_df, args.out_path)
    only_correct = remove_incorrect_lines(processed_df)
    save_df(only_correct, args.out_path)
    save_ids_of_analyze(only_correct["sentence_id"].values, args.out_path)
    plot_cdf(only_correct, args.out_path, "Wino")
