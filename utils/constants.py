import os.path

# PATHS
DATA_DIR = "original_data"
BUG_ORIGINAL_DATASET_PATH = os.path.join(DATA_DIR, "gold_BUG.csv")
WINO_DATASET_PATH = os.path.join(DATA_DIR, "wino_combined.csv")
WINOGENDER_ORIGINAL_PATH = os.path.join(DATA_DIR, "winogender_original.tsv")

EXPERIMENT_RESULTS_DIR = "experiment_results"
RAW_EXPERIMENT_RESULTS_DIR = os.path.join(EXPERIMENT_RESULTS_DIR, "raw")
MODELS_RAW_RESULTS_DIR = os.path.join(RAW_EXPERIMENT_RESULTS_DIR, "models")
HUMANS_RAW_RESULTS_DIR = os.path.join(RAW_EXPERIMENT_RESULTS_DIR, "humans")
QA_HUMANS_RAW_RESULTS_DIR = os.path.join(HUMANS_RAW_RESULTS_DIR, "QA")
MAZE_HUMANS_RAW_RESULTS_DIR = os.path.join(HUMANS_RAW_RESULTS_DIR, "MAZE")

PROCESSED_EXPERIMENT_RESULTS_DIR = os.path.join(EXPERIMENT_RESULTS_DIR, "processed")
HUMANS_PROCESSED_RES_DIR = os.path.join(PROCESSED_EXPERIMENT_RESULTS_DIR, "humans")
PROCESSED_QA_RES_DIR = os.path.join(HUMANS_PROCESSED_RES_DIR, "QA")

ENROLLMENT_QA_DIR = os.path.join(EXPERIMENT_RESULTS_DIR, "enrollment_QA")


# DEFAULT_OUT_PATHS
DEFAULT_MODELS_RESULT_PATH = os.path.join(PROCESSED_EXPERIMENT_RESULTS_DIR, "models_eval.json")


# Seperator for parsing csv file to pandas
TABLE_SEPARATOR = '\t'

# According to our suggested fine-grained categorization, we suggest a more specific
# division to pro and anti stereotype samples:
ANTI_STEREOTYPE_SENTENCES_TYPES = ['anti-pro', 'neutral-pro', 'anti-neutral']
PRO_STEREOTYPE_SENTENCES_TYPES = ['pro-anti', 'neutral-anti', 'pro-neutral']


DATASET_NAMES = ["BUG", "wino"]
MODEL_NAMES = ["SpanBERT", "s2e"]

