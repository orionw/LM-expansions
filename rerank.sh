#!/bin/bash
dataset_name=$1
file_append=$2
# whether to use chunks if desired
chunk_idx=$3
chunk_queries=$4
# the combs are the expansions
doc_comb=$5
# the types are "append", "replace", etc. how you combine them
doc_comb_type=$6
query_comb=$7
query_comb_type=$8

# check if $9 (the chosen model) is equal to inpars or monot5 or trained
if [[ "${9}" == "inpars" ]]; then
    chosen_model="zeta-alpha-ai/monot5-3b-inpars-v2-$(echo $dataset_name | tr '-' '_')"
elif [[ "${9}" == "bm25" ]]; then
    chosen_model="bm25"
elif [[ "${9}" == "monot5" ]]; then
    chosen_model="castorini/monot5-3b-msmarco-10k"
elif [[ "${9}" == "monot5-small" ]]; then
    chosen_model="castorini/monot5-small-msmarco-10k"
elif [[ "${9}" == "tart" ]]; then
    chosen_model="facebook/tart-full-flan-t5-xl"
elif [[ "${9}" == "flan" ]]; then
    chosen_model="google/flan-t5-xl"
elif [[ "${9}" == "wizard" ]]; then
    chosen_model="conceptofmind/Flan-Open-Llama-7b"
elif [[ "${9}" == "contriever" ]]; then
    chosen_model="facebook/contriever"
elif [[ "${9}" == "contriever_msmarco" ]]; then
    chosen_model="facebook/contriever-msmarco"
elif [[ "${9}" == "monobert" ]]; then
    chosen_model="castorini/monobert-large-msmarco-finetune-only"
elif [[ "${9}" == "dpr" ]]; then
    chosen_model="facebook/dpr-question_encoder-single-nq-base"
elif [[ "${9}" == "colbertv2" ]]; then
    chosen_model="colbertv2.0"
elif [[ "${9}" == "spladev2" ]]; then
    chosen_model="naver/splade_v2_distil"
elif [[ "${9}" == "monot5-base" ]]; then
    chosen_model="castorini/monot5-base-msmarco-10k"
elif [[ "${9}" == "monot5-large" ]]; then
    chosen_model="castorini/monot5-large-msmarco-10k"
elif [[ "${9}" == "ms-marco-MiniLM-L-4-v2" ]]; then
    chosen_model="cross-encoder/ms-marco-MiniLM-L-4-v2"
elif [[ "${9}" == "ms-marco-MiniLM-L-2-v2" ]]; then
    chosen_model="cross-encoder/ms-marco-MiniLM-L-2-v2"
elif [[ "${9}" == "ms-marco-MiniLM-L-12-v2" ]]; then
    chosen_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
elif [[ "${9}" == "e5-large-v2" ]]; then
    chosen_model="intfloat/e5-large-v2"
elif [[ "${9}" == "gte-large" ]]; then
    chosen_model="thenlper/gte-large"
elif [[ "${9}" == "bge-large-en" ]]; then
    chosen_model="BAAI/bge-large-en"
elif [[ "${9}" == "e5-small" ]]; then
    chosen_model="intfloat/e5-small-v2"
elif [[ "${9}" == "e5-base-v2" ]]; then
    chosen_model="intfloat/e5-base-v2"
elif [[ "${9}" == "gte-small" ]]; then
    chosen_model="thenlper/gte-small"
elif [[ "${9}" == "bge-small" ]]; then
    chosen_model="BAAI/bge-small-en"
elif [[ "${9}" == "all-mpnet-base-v2" ]]; then
    chosen_model="sentence-transformers/all-mpnet-base-v2"
elif [[ "${9}" == "llama2" ]]; then
    chosen_model="orionweller/llama2-reranker-msmarco-7b"
elif [[ "${9}" == "llama2-13b" ]]; then
    chosen_model="orionweller/llama2-reranker-msmarco-13b"
elif [[ "${9}" == "custom" ]]; then
    chosen_model="${13}"
else
    echo "Error: ${9} is not a valid model name. Please see the options in 'rerank.sh'."
    exit 1
fi

max_queries=${10}
max_docs=${11}

echo "Reranking with $chosen_model..."

dataset_name_slash_to_dash=$(echo $1 | sed 's/\//--/g')

# make results/$dataset_name_slash_to_dash-$file_append/ if not exists
mkdir -p -m 777 results/$dataset_name_slash_to_dash
mkdir -p -m 777 results/$dataset_name_slash_to_dash/$file_append/


if [[ $chosen_model == "bm25_full" ]]; then
    echo "Running bm25 full"
    SAVE_PATH="results/$dataset_name_slash_to_dash/$file_append/$dataset_name_slash_to_dash-$file_append-run.txt"
    echo "bash make_bm25_run.sh $SAVE_PATH $dataset_name doc_id title,text query_id text $query_comb $doc_comb"
    bash make_bm25_run.sh $SAVE_PATH $dataset_name doc_id title,text query_id text $query_comb $doc_comb
    
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "success"
    else
        exit 1
    fi
    exit 0
fi

python -m expansions.rerank \
        --model $chosen_model \
        --dataset $dataset_name \
        --output_run "results/$dataset_name_slash_to_dash/$file_append/$dataset_name_slash_to_dash-$file_append-run.txt" \
        --batch_size 16 \
        --chunk_idx $chunk_idx \
        --chunk_queries $chunk_queries \
        --fp16 \
        --corpus $doc_comb \
        --corpus_combination $doc_comb_type \
        --queries $query_comb \
        --queries_combination $query_comb_type \
        --max_num_queries $max_queries \
        --max_num_docs $max_docs \
        --torch_compile 


# example usage:
#   bash rerank.sh "scifact_refute" "testing" 0 1 "none" "none" "none" "none" "facebook/contriever-msmarco" 10 100
