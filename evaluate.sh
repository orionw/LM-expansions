#!/bin/bash
dataset_name=$1
file_append=$2 # needs to match what was in rerank


dataset_name_slash_to_dash=$(echo $1 | sed 's/\//--/g')

python -m expansions.evaluate \
        --dataset=$dataset_name \
        --run="results/$dataset_name_slash_to_dash/$file_append/$dataset_name_slash_to_dash-$file_append-run.txt" \
        --json 
        

# example usage:
#       bash evaluate.sh scifact_refute testing 
