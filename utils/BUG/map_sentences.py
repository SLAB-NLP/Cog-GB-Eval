

from utils.constants import BUG_ORIGINAL_DATASET_PATH
import pandas as pd
import json
from tqdm import tqdm
# js_file = 'data/BUG/MAZE/sample.js'
#
#
# def main():
#     orig_data = pd.read_csv(BUG_ORIGINAL_DATASET_PATH, encoding='latin-1')
#     with open(js_file) as f:
#         lines = f.readlines()
#     lines = [line.strip() for line in lines if line.startswith("[[\"male") or line.startswith("[[\"female")]
#     mapping = {}
#     for line in tqdm(lines):
#         start = line.find("{s:")
#         end = line.find(", a:")
#         sentence = line[start+4: end-1]
#         crop_index = sentence.find('#')
#         key_sentence = sentence if crop_index == -1 else sentence[:crop_index]
#         # print(sentence)
#         df = orig_data[orig_data["sentence_text"] == sentence]
#         if len(df) > 0:
#             original_row = df.iloc[0]
#             mapping[key_sentence] = int(original_row["uid"])
#         else:
#             max_uid = get_closest_sentence(orig_data, sentence)
#             mapping[key_sentence] = int(max_uid)
#     out = generate_script_out_path("MAZE_BUG_mapping", "map.json")
#     with open(out, 'wt') as f:
#         json.dump(mapping, f)


def get_closest_sentence(orig_data, sentence):
    max_uid = -999
    max_intersect = 0
    sent_split = sentence.split(' ')
    for i, row in orig_data.iterrows():
        if row.isnull()["sentence_text"]:
            continue
        intersect = set(row['sentence_text'].split(' ')) & set(sent_split)
        if len(intersect) > max_intersect:
            max_uid = row["uid"]
            max_intersect = len(intersect)
    assert max_uid != -999
    return max_uid


if __name__ == '__main__':
    main()
