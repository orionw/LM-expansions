import os
import csv
import requests
import pandas as pd
from tqdm.auto import tqdm

PREBUILT_RUN_URL = "https://huggingface.co/datasets/orionweller/ir-runs/resolve/main/bm25/run.beir-v1.0.0-{dataset}-flat.trec"
RUNS_CACHE_FOLDER = os.environ["HOME"] + "/.cache/inpars"

BEIR_DATASETS = [
    "bioasq",
    "scifact",
    "trec-covid",
    "hotpotqa",
    "nq",
    "fiqa",
    "arguana",
    "nfcorpus",
    "signal1m"
    "trec-news",
    "robust04",
    "quora",
    "fever",
    "climate-fever",
    "scidocs",
    "cqadupstack",
    "dbpedia-entity",
    "webis-touche2020",
    "msmarco",
    "trec_covid"
]

def is_tsv_dataset(dataset_name):
    datasets_tsv = ["codesearchnet/challenge"]
    for dataset in datasets_tsv:
        if dataset in dataset_name:
            return True
    return False

# https://stackoverflow.com/a/62113293
def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class TRECRun:
    def __init__(self, run_file, sep=r"\s+", num_queries: int = None, num_docs: int = None):
        if not os.path.exists(run_file):
            if "/" in run_file:
                if "cqadupstack" in run_file:
                    run_file = run_file.replace("/", "-")
                else:
                    run_file = run_file.replace("/", "--")
                
            dest_file = os.path.join(
                RUNS_CACHE_FOLDER,
                "runs",
                "run.beir-v1.0.0-{dataset}-flat.trec".format(dataset=run_file),
            )
            if not os.path.exists(dest_file):
                os.makedirs(os.path.dirname(os.path.abspath(dest_file)), exist_ok=True)
                # TODO handle errors ("Entry not found")
                download(PREBUILT_RUN_URL.format(dataset=run_file), dest_file)
            run_file = dest_file

        self.run_file = run_file
        print("Loading run file: ", run_file)
        if not is_tsv_dataset(run_file):
            self.df = pd.read_csv(
                run_file,
                sep=sep,
                quoting=csv.QUOTE_NONE,
                keep_default_na=False,
                names=("qid", "_1", "docid", "rank", "score", "ranker"),
                dtype=str,
            )
            self.df["rank"] = self.df["rank"].astype(int)
            self.df = self.df.sort_values(["qid", "rank"])
        else:
            print("Loading TSV dataset")
            self.df = pd.read_csv(
                run_file,
                sep="\t",
                keep_default_na=False,
                names=("qid", "docid", "rank"),
                dtype=str,
            )
            self.df["score"] = -1
            self.df["_1"] = "Q0"
            self.df["ranker"] = "unknown"
            self.df = self.df[["qid", "_1", "docid", "rank", "score", "ranker"]]
            self.df["rank"] = self.df["rank"].astype(int)
            self.df = self.df.sort_values(["qid", "rank"])

        if num_queries:
            self.df = self.df.sort_values(["qid", "rank"])
            first_n_queries = self.df["qid"].drop_duplicates()[:num_queries].tolist()
            self.df = self.df[self.df["qid"].isin(first_n_queries)]
            assert self.df.shape[0] > 1, "Run is empty"
            print(self.df.qid.unique().tolist()[:10])
        
        if num_docs: 
            self.df = self.df.sort_values(["qid", "rank"])
            all_queries = []
            for query_id in tqdm(self.df.qid.drop_duplicates().tolist()):
                df_query = self.df[self.df.qid == query_id]
                df_top_ranked = df_query[df_query["rank"].astype(int) <= num_docs]
                all_queries.append(df_top_ranked)
            self.df = pd.concat(all_queries)

    def rerank(self, ranker, queries, corpus, top_k=1000):
        # Converts run to float32 and subtracts a large number to ensure the BM25 scores
        # are lower than those provided by the neural ranker.
        if "score" in self.df.columns:
            self.df["score"] = (
                self.df["score"]
                .astype("float32")
                .apply(lambda x: x-10000)
            )
        else:
            self.df["score"] = -1

        # Only keep rows in the current query chunk
        print(f"DF is {self.df.shape[0]} rows")
        self.df = self.df[self.df["qid"].isin(queries)]

        # Reranks only the top-k documents for each query
        subset = (
            self.df[["qid", "docid"]]
            .groupby("qid")
            .head(top_k)
            .apply(lambda x: [queries[x["qid"]], corpus[x["docid"]]], axis=1)
        )

        scores = ranker.rescore(subset.values.tolist())
        
        self.df.loc[subset.index, "score"] = scores

        self.df["ranker"] = ranker.name
        self.df = (
            self.df
            .groupby("qid")
            .apply(lambda x: x.sort_values("score", ascending=False))
            .reset_index(drop=True)
        )

        self.df["rank"] = self.df.groupby("qid").cumcount() + 1

    def save(self, path):
        self.df.to_csv(path, index=False, sep="\t", header=False, float_format='%.15f')
