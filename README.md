# Cog-GB-Eval
Compare humans gender bias and coreference resolution models. Lior &amp; Stanovsky, CogSci 2023

### Model results parsing

From jsonl raw results file to a list of correct and incorrect sentence ids:

`analyze_models/generate_result_ids_lists.py --out_path <path> --model_name s2e --model_name SpanBERT`
