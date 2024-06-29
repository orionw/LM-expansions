#!/bin/bash
dataset_name=$1
prompt_config=$2
python -m expansions.batch_predict -f $prompt_config

# example usage:
#   bash generate_expansions.sh "scifact_refute" "prompt_configs/chatgpt_doc2query.jsonl"