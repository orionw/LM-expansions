#!/bin/bash
base_path="/nfs"
cd $base_path/beir-runs
save_path=$1
dataset_id=$2
# fields can be combined with "," for text fields
id_key=$3
combined_doc_cols=$4
query_id_key=$5
combined_query_cols=$6
# these are optional
path_to_queries=$7
path_to_docs=$8

small_dataset_id=$(basename $dataset_id)
file_safe_dataset_id=$(echo $dataset_id | sed 's/\//--/g')

IR_DATASET_NAME=$dataset_id
# if dataset name == fiqa, add beir/*/test to the front else if arguana add beir
if [ $dataset_id == "fiqa" ]
then
    IR_DATASET_NAME="beir/$dataset_id/test"
elif [ $dataset_id == "arguana" ]
then
    IR_DATASET_NAME="beir/$dataset_id"
fi
## if local 
mkdir -p ${file_safe_dataset_id}_docs

# if the path to docs is empty or equal to none
if [ -z "$path_to_docs" ] || [ "$path_to_docs" == "none" ]
then
    echo "path_to_docs is empty"
    ir_datasets export $IR_DATASET_NAME docs --format jsonl > ${file_safe_dataset_id}_docs/docs.jsonl
else
    echo "path_to_docs is not empty"
    cat /nfs/InPars/$path_to_docs > ${file_safe_dataset_id}_docs/docs.jsonl
    head -n 1 ${file_safe_dataset_id}_docs/docs.jsonl
    id_key="doc_id"
fi

# if the path to queries is empty or equal to "none"
if [ -z "$path_to_queries" ] || [ "$path_to_queries" == "none" ]
then
    echo "path_to_queries is empty"
    ir_datasets export $IR_DATASET_NAME queries --format jsonl > ${file_safe_dataset_id}_queries.jsonl
else
    echo "path_to_queries is not empty"
    cat /nfs/InPars/$path_to_queries > ${file_safe_dataset_id}_queries.jsonl
    head -n 1 ${file_safe_dataset_id}_queries.jsonl
    query_id_key="_id"
fi

cp ${file_safe_dataset_id}_queries.jsonl ${file_safe_dataset_id}_queries_og.jsonl

# check if combined_query_cols is not empty
if [ -n "$combined_query_cols" ]
then
    echo "combined_query_cols is not empty"
    # separate query columns by ","
    IFS=',' read -ra query_cols <<< "$combined_query_cols"

    # if there's only one element then don't use parens
    only_one=${#query_cols[@]}
    if [ $only_one -eq 1 ]
    then
        jq_cmd='{query_id: .'$query_id_key
        jq_cmd+=', contents: .'${query_cols[0]}' }'
        echo $jq_cmd
        jq "$jq_cmd" ${file_safe_dataset_id}_queries.jsonl | jq -c . > ${file_safe_dataset_id}_queries1.jsonl
    else
        jq_cmd='{query_id: .'$query_id_key
        jq_cmd+=', contents: ('
        for i in "${!query_cols[@]}"; do
            if [ $i -eq 0 ]
            then
                jq_cmd+='.'${query_cols[$i]}
            else
                jq_cmd+=' // "" + " " + .'${query_cols[$i]}
            fi
        done
        jq_cmd+=' // "" )}'
        echo $jq_cmd
        # | jq -c .[]
        jq "$jq_cmd" ${file_safe_dataset_id}_queries.jsonl   > ${file_safe_dataset_id}_queries1.jsonl
    fi


    cp ${file_safe_dataset_id}_queries1.jsonl ${file_safe_dataset_id}_queries2.jsonl
    mv ${file_safe_dataset_id}_queries1.jsonl ${file_safe_dataset_id}_queries.jsonl

    # convert json lines file to tsv using jq
    jq -r '[.query_id, .contents] | @tsv' ${file_safe_dataset_id}_queries.jsonl > ${file_safe_dataset_id}_queries.tsv
fi


# check if combined_doc_cols key is not empty
if [ -n $combined_doc_cols ]
then
    echo "combined_doc_cols is not empty"
    # separate doc columns by ","
    IFS=',' read -ra doc_cols <<< "$combined_doc_cols"

    # if there's only one element then don't use parens
    only_one=${#doc_cols[@]}
    if [ $only_one -eq 1 ]
    then
        jq_cmd='{id: .'$id_key
        jq_cmd+=', contents: .'${doc_cols[0]}
        jq_cmd+='}'
        echo $jq_cmd
        jq "$jq_cmd" ${file_safe_dataset_id}_docs/docs.jsonl | jq -c . > ${file_safe_dataset_id}_docs/docs1.jsonl
    else
        jq_cmd='{id: .'$id_key
        jq_cmd+=', contents: ('
        for i in "${!doc_cols[@]}"; do
            if [ $i -eq 0 ]
            then
                jq_cmd+='(.'${doc_cols[$i]}
            else
                jq_cmd+=' // "") + " " + (.'${doc_cols[$i]}
            fi
        done
        jq_cmd+=' // ""))}'
        echo $jq_cmd
        jq "$jq_cmd" ${file_safe_dataset_id}_docs/docs.jsonl > ${file_safe_dataset_id}_docs/docs1.jsonl
    fi
    mv ${file_safe_dataset_id}_docs/docs1.jsonl ${file_safe_dataset_id}_docs/docs.jsonl
fi

mkdir -p indexes/${file_safe_dataset_id}/

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${file_safe_dataset_id}_docs \
  --index indexes/${file_safe_dataset_id}/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1

head ${file_safe_dataset_id}_queries.tsv

python -m pyserini.search.lucene \
  --index indexes/${file_safe_dataset_id}/ \
  --topics ${file_safe_dataset_id}_queries.tsv \
  --output /nfs/InPars/$save_path \
  --bm25 \
  --hits 2000 \
  --output-format trec

# example usage:
#    bash make_bm25_run.sh bm25 scifact_refute doc_id "title,text" query_id text
