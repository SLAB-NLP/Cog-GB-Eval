# original_data
This directory includes the original datasets used for gender bias evaluation.

This directory contains the following files:
- `wino_combined.csv`: A combined version of winobias and winogender datasets. This csv is a processed version of the original datasets, includes additional columns used for our evaluations.
- `gold_BUG.csv`: This is a partial version of the original BUG dataset, including only sentences that were manually annotated for stereotype label (pro/anti/neutral).
- `BUG_ids_mapping.json`: Mapping between sentences and their id, according to the csv.
- `winogender_original.tsv`: The original version of only winogender sentences.

### References
####BUG
Paper: https://arxiv.org/pdf/2109.03858.pdf

Git repository: https://github.com/SLAB-NLP/BUG

####winogender
Paper: https://aclanthology.org/N18-2002.pdf

Git repository: https://github.com/rudinger/winogender-schemas

####winobias
Paper: https://arxiv.org/pdf/1804.06876.pdf

Git repository: https://github.com/uclanlp/corefBias
