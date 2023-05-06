# Cog-GB-Eval
Compare humans gender bias and coreference resolution models. Lior &amp; Stanovsky, CogSci 2023

###Model results parsing

From jsonl raw results file to a list of correct and incorrect sentence ids:

`analyze_models/generate_result_ids_lists.py --out_path <path> --model_name s2e --model_name SpanBERT`

todo:
- add models points on maze graphs
- document all existing scripts (also to directories)
- write a bash script that reconstruct our results
- add explanation on how to add a new model results