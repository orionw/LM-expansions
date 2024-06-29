import os
import json
import argparse
import pandas as pd
from .dataset import load_corpus, load_queries
from .evaluate import load_qrel_file

def read_in_file(file_path: str):
    qrels_to_write = []
    with open(file_path, "r") as fin:
        for line in fin:
            qrels_to_write.append(line)
    return qrels_to_write

def get_stats(args):
    for dataset in args.datasets:
        corpus = load_corpus(dataset)
        queries = load_queries(dataset)
        qrels_file = load_qrel_file(dataset)
        qrels = read_in_file(qrels_file)

        print(f"Dataset: {dataset}")
        corpus_size = len(corpus)
        query_size = len(queries)
        ave_doc_q = len(qrels) / len(queries)
        ave_query_len = sum([len(queries[q].split(" ")) for q in queries]) / query_size
        if type(corpus) == pd.DataFrame:
            ave_doc_len = corpus.text.apply(lambda x: len(x.split(" "))).mean()
        elif type(corpus) == dict:
            ave_doc_len = sum([len(corpus[d[list(d.keys())[0]]].split(" ")) for d in corpus]) / corpus_size
        else:
            breakpoint()

        latex_str = f"& {query_size:,} & {corpus_size:,} & {ave_doc_q:.1f} & {ave_query_len:.1f} & {ave_doc_len:.1f} \\\\"
        print(latex_str)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datasets", type=str, nargs="+", required=True)
    args = parser.parse_args()

    get_stats(args)
    # python -m expansions.dataset_info -d msmarco-passage/trec-dl-2019/judged msmarco-passage/trec-dl-2020/judged fiqa gooaq_technical nfcorpus webis-touche2020 scifact_refute fever_refute tot clinicaltrials/2021/trec-ct-2021 arguana ambig linkso_py quora
