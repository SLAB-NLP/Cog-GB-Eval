
from utils.constants import PATH_TO_MODELS_RESULTS, ALL_DATA_PATH, TABLE_SEPARATOR, \
    BUG_ORIGINAL_DATASET_PATH, PRO_STEREOTYPE_SENTENCES_TYPES

import os
from tqdm import tqdm
import jsonlines
import json
import pandas as pd
from itertools import product
from utils.BUG.map_sentences import get_closest_sentence
from utils.path_generation import generate_script_out_path


def eval_single_model(dataset, model, original_dataset):
    path_to_res = os.path.join(PATH_TO_MODELS_RESULTS,
                               f"{dataset.lower()}_{model.lower()}.jsonl")
    with jsonlines.open(path_to_res) as f:
        correct = []
        incorrect = []
        for line in tqdm(f.iter()):
            if "tokens" in line:
                tokens = line["tokens"]
                if "sentence" in line:
                    sentence = line["sentence"]
                else:
                    sentence = line["sentence_text"]
            else:
                tokens = line["document"]
                sentence = ' '.join(tokens)
            clusters = line["clusters"]
            if dataset == 'BUG':
                uid = get_closest_sentence(original_dataset, sentence)
                original_row = original_dataset[original_dataset["uid"] == uid].iloc[0]
                main_entity = original_row["profession"]
                pronoun = original_row["g"]
            else:
                sentence = sentence.replace(' .', '.').replace('can not', 'cannot').\
                    replace(' ,', ',').replace(' n\'t', 'n\'t').replace(' - ', '-').\
                    replace(', 0', ',0').replace(' ;', ';').replace('don\'thing', 'do nothing')
                if sentence not in set(original_dataset["sentence"]):
                    sentence = sentence.replace(' \'s', '\'s')
                # print(sentence)
                # print(original_dataset.iloc[2913])
                original_row = original_dataset[original_dataset["sentence"] == sentence].iloc[0]
                main_entity = original_row["main_entity_occupation"]
                pronoun = original_row["pronoun"]
                uid = original_row.name
                assert original_row.name == original_row["Unnamed: 0"]
            if len(clusters) == 0:
                incorrect.append(int(uid))
                # print(line)
                continue
            for cluster in clusters:
                c_pronoun, c_entity = False, False
                for node in cluster:
                    words_in_cluster = ' '.join(tokens[node[0]:node[1]+1])
                    if main_entity in words_in_cluster:
                        c_entity = True
                    if pronoun in words_in_cluster:
                        c_pronoun = True
                if c_pronoun and c_entity:
                    correct.append(int(uid))
                    break
            else:
                incorrect.append(int(uid))
        return correct, incorrect


def main():
    all_results = {}
    for dataset, model in product(["BUG", "Wino"], ["SpanBERT","Kirstain"]):
        print(dataset, model)
        if dataset == 'BUG':
            original_dataset = pd.read_csv(BUG_ORIGINAL_DATASET_PATH, encoding='latin-1')
        else:
            original_dataset = pd.read_csv(ALL_DATA_PATH, TABLE_SEPARATOR)
            # with open("scripts/output/unique/220609_190329_ids_winobias_QA.pkl", 'rb') as f:
            #     wino_bias_ids = pickle.load(f)
            # with open("scripts/output/unique/220609_190329_ids_winogender_QA.pkl", 'rb') as f:
            #     wino_gender_ids = pickle.load(f)
            # print(wino_bias_ids, wino_gender_ids)
            # same_as_humans = original_dataset.iloc[list(wino_bias_ids.union(wino_gender_ids))]
            original_dataset["stereotype"] = original_dataset["type"].apply(lambda t: t in PRO_STEREOTYPE_SENTENCES_TYPES)
        correct, incorrect = eval_single_model(dataset, model, original_dataset)
        print(len(correct), len(incorrect))
        all_results[f"{dataset}-{model}"] = {"correct": correct, "incorrect": incorrect}
    out_path = generate_script_out_path("models_eval", "eval.json", unique_date=True)
    with open(out_path, 'wt') as f:
        json.dump(all_results, f)


if __name__ == '__main__':

    main()
