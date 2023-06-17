# Cognitive Gender Bias Evaluation

Supporting repo for [Comparing Humans and Models on a Similar Scale: Towards Cognitive Gender Bias Evaluation in Coreference Resolution](https://arxiv.org/abs/2305.15389).

[<em>Gili Lior</em>](https://gililior.github.io), [<em>Gabriel Stanovsky</em>](https://gabrielstanovsky.github.io/)

CogSci 2023


## Reproducing our results

Tested on python 3.7.


### Model results parsing

From jsonl raw results file to a list of correct and incorrect sentence ids:

`python analyze_models/generate_result_ids_lists.py --out_path --out_path experiment_results/processed/models_eval.json --model_name s2e --model_name SpanBERT`


###QA results parsing
Generate QA results table from humans raw csv results.

BUG:

`python analysis_scripts/QA/generate_human_results_table.py --dataset BUG --out_path experiment_results/processed/humans/QA/BUG`

Wino:

`python analysis_scripts/QA/generate_human_results_table.py --dataset wino --out_path experiment_results/processed/humans/QA/wino`

### MAZE results parsing

Generate plots from humans raw results over the MAZE task.

BUG:

`python analysis_scripts/MAZE/BUG_plots.py --out_path experiment_results/processed/humans/MAZE/BUG`

Wino:

`python analysis_scripts/MAZE/wino_plots.py --out_path experiment_results/processed/humans/MAZE/wino`

DELTA plots (difference between pro and anti stereotype for both datasets):

`python analysis_scripts/MAZE/delta_plots.py --results_dir experiment_results/processed/humans/MAZE`


###Compare human and model results

####QA
Generate a combined table results:

`python analysis_scripts/QA/generate_combined_results.py --models_path experiment_results/processed/models/eval.json --humans_path experiment_results/processed/humans/QA/ --out experiment_results/processed/final/QA_combined_table.md`

Plot the results on a graph:

`python analysis_scripts/QA/plot_results.py --in_results_md experiment_results/processed/QA_combined_table.md --out experiment_results/processed/QA_plot.png`


Qualitative analysis:

`python analysis_scripts/QA/qualitative_analysis.py --models_path experiment_results/processed/models/eval.json --humans_path experiment_results/processed/humans/QA --out experiment_results/processed/final`

####MAZE

Generate a combined plot for humans and models, one for wino and one for BUG:

`python analysis_scripts/MAZE/with_models_plot.py --human_results experiment_results/processed/humans/MAZE --models_path experiment_results/processed/models/eval.json --dataset wino --out experiment_results/processed/final`

`python analysis_scripts/MAZE/with_models_plot.py --human_results experiment_results/processed/humans/MAZE --models_path experiment_results/processed/models/eval.json --dataset BUG --out experiment_results/processed/final`


## Directory mapping
[analysis_scripts](analysis_scripts):
This directory conatains all the code for analyzing and comparing human and model results.

[experiment_results](experiment_results): Contains all the results.
[experiment_results/raw](experiment_results/raw): raw results of the two human evaluation tasks, as well as models raw results.
[experiment_results/processed](experiment_results/processed): results of all scripts from [analysis_scripts](analysis_scripts).

[original_data](original_data): The original datasets (winogender, winobias, gold BUG), as well as some metadata. 


##Evaluating a new model
In order to use the above script to analyze a new coreference model, the following steps need to be maid

1. Run your model over [BUG](original_data/gold_BUG.csv) and [wino](original_data/wino_combined.csv) datasets.
2. In order to run the model results processing [script](analyze_models/generate_result_ids_lists.py), you will need to generate a jsonl file for each dataset results. See format example [here](experiment_results/raw/models/s2e/BUG.jsonl).
3. Place your jsonl files (one for wino, one for BUG) under [experiment_results/raw/models](experiment_results/raw/models)/<your-model-name>.
4. Run [analyze_models/generate_result_ids_lists.py](analyze_models/generate_result_ids_lists.py) with the option "--model <your-model-name>", as well as all other necessary options as mentioned above.
5. Run all the scripts from the 'Compare human and model results' part.