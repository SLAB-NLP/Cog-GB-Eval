import os.path

# PATHS
DATA_DIR = "original_data"
BUG_ORIGINAL_DATASET_PATH = os.path.join(DATA_DIR, "gold_BUG.csv")
WINO_DATASET_PATH = os.path.join(DATA_DIR, "wino_combined.csv")
WINOGENDER_ORIGINAL_PATH = os.path.join(DATA_DIR, "winogender_original.tsv")
BUG_IDS_MAPPING_PATH = os.path.join(DATA_DIR, "BUG_ids_mapping.json")
US_LABOR_PATH = os.path.join(DATA_DIR, "cpsaat11.csv")
WINO_STATS_PATH = os.path.join(DATA_DIR, "occupations-stats.tsv")

EXPERIMENT_RESULTS_DIR = "experiment_results"
RAW_EXPERIMENT_RESULTS_DIR = os.path.join(EXPERIMENT_RESULTS_DIR, "raw")
MODELS_RAW_RESULTS_DIR = os.path.join(RAW_EXPERIMENT_RESULTS_DIR, "models")
HUMANS_RAW_RESULTS_DIR = os.path.join(RAW_EXPERIMENT_RESULTS_DIR, "humans")
QA_HUMANS_RAW_RESULTS_DIR = os.path.join(HUMANS_RAW_RESULTS_DIR, "QA")
MAZE_HUMANS_RAW_RESULTS_DIR = os.path.join(HUMANS_RAW_RESULTS_DIR, "MAZE")
ENROLLMENT_QA_DIR = os.path.join(HUMANS_RAW_RESULTS_DIR, "enrollment_QA")


# Seperator for parsing wino csv file to pandas
TABLE_SEPARATOR = '\t'


# According to our suggested fine-grained categorization, we suggest a more specific
# division to pro and anti stereotype samples:
ANTI_STEREOTYPE_SENTENCES_TYPES = ['anti-pro', 'neutral-pro', 'anti-neutral']
PRO_STEREOTYPE_SENTENCES_TYPES = ['pro-anti', 'neutral-anti', 'pro-neutral']


# If you add any more models / dataset results, you should update its name here, and
# locate the data in the matching directories according to the existing models and
# datasets.
DATASET_NAMES = ["BUG", "wino"]
MODEL_NAMES = ["SpanBERT", "s2e"]

