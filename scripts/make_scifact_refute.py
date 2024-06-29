import os
import glob
import pandas as pd
import datasets
import json

## make sure to download and unzip `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`

# load HF dataset version of scifact/validation
scifact = datasets.load_dataset("allenai/scifact", "claims")
# convert DatasetDict to pandas
scifact_df = scifact["validation"].to_pandas()
# get all the queries that are CONTRADICT
contradict_queries = scifact_df[scifact_df["evidence_label"] == "CONTRADICT"]["id"].unique()

print(f"Found {len(contradict_queries)} contradict queries")

BASE_SCIFACT_PATH = "scifact/"

# subset the queries.jsonl file to only contain contradict queries
queries = []
with open(BASE_SCIFACT_PATH + "queries.jsonl", "r") as f:
    for line in f:
        queries.append(json.loads(line))

contradict_queries = set(contradict_queries)
# filter by id in contradict_queries
queries = [q for q in queries if int(q["_id"]) in contradict_queries]
# write to new file
with open(BASE_SCIFACT_PATH + "contradict_queries.jsonl", "w") as f:
    for q in queries:
        f.write(json.dumps(q) + "\n")

# subset the qrels with only the ones that are CONTRADICT
qrel_df = pd.read_csv(BASE_SCIFACT_PATH + "qrels/test.tsv", sep="\t", header=0)
# subset on the query-id column
qrel_df = qrel_df[qrel_df["query-id"].astype(int).isin(contradict_queries)]
print(f"Found {qrel_df.shape[0]} contradict qrels")
# write to new file
qrel_df.to_csv(BASE_SCIFACT_PATH + "contradict_qrels.tsv", sep="\t", index=False)



