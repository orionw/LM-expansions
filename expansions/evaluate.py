import os
import json
import argparse
import subprocess
from pyserini.search import get_qrels_file
from .utils import TRECRun, BEIR_DATASETS
from .dataset import make_qrels_file, LOCAL_DATASETS, PATH_TO_LOCAL


def load_qrel_file(dataset: str, qrels: str = None):
    if dataset == "msmarco":
        dataset_name = "msmarco-passage-dev-subset"
    elif dataset in BEIR_DATASETS or "cqadupstack" in dataset:
        if "cqadupstack" in dataset:
            dataset = dataset.replace("/", "-")
        dataset_name = f"beir-v1.0.0-{dataset}-test"
    else:
        dataset_name = dataset

    try:
        qrels_file = get_qrels_file(dataset_name)
    except FileNotFoundError as e: # may not be in pyserini
        print(e)
        if dataset_name in LOCAL_DATASETS:
            qrel_path = f"{PATH_TO_LOCAL}/{dataset_name}/qrels/test.tsv"
            os.system(f"cat {qrel_path} | tail -n +2 > {qrel_path}.tmp")
            # add a new column to index 1 with value 0 using the command sed, pushing the other columns one back
            os.system(f"cat {qrel_path}.tmp | sed 's/\\t/\\t0\\t/' > {qrel_path}.tmp.2")
            qrels_file = qrel_path + ".tmp.2"
        else:
            qrels_file = make_qrels_file(dataset)

    if qrels and os.path.exists(qrels):
        qrels_file = qrels

    return qrels_file

def filter_by_run(run_file, qrels_file):
    run = TRECRun(run_file)
    queries = run.df.qid.astype(str).unique().tolist()

    # read in qrels file
    qrels_to_write = []
    skip_lines = 0
    with open(qrels_file, "r") as fin:
        for line in fin:
            qid, _, docid, rel = line.strip().split()
            if str(qid) not in queries:
                print("Missing ID for QID", qid)
                skip_lines += 1
                continue
            qrels_to_write.append(line)

    qrels_file = str(qrels_file)
    with open(qrels_file + ".filtered", "w") as fout:
        fout.write("".join(qrels_to_write))

    print(f"Filtered qrels file written to {qrels_file}.filtered")
    print(f"Skipped {skip_lines} lines from {qrels_file}")
        
    return run_file, qrels_file + ".filtered"



def run_trec_eval(run_file, qrels_file, relevance_threshold=1, remove_unjudged=False, rr: bool = False):
    args = [
        "python3",
        "-m",
        "pyserini.eval.trec_eval",
        "-q",
        "-c",
    ]
    print(rr, type(rr))
    if rr:
        args += [
            "-M",
            "10",
            "-m",
            "recip_rank"
        ]
    else:
        args += [
        f"-l {relevance_threshold}",
        "-m" , "all_trec",
        "-m", "judged.10",
        "-m", "recall.5,10,15,20,30,100,200,500,1000",
        "-m", "ndcg_cut.5,10,15,20,30,100,200,500,1000",
    ]

    if remove_unjudged:
        args.append("-remove-unjudged")
    args += [qrels_file, run_file]
    print(args)

    result = subprocess.run(args, stdout=subprocess.PIPE)
    print(result)
    metrics = {}
    for line in result.stdout.decode("utf-8").split("\n"):
        for metric in [
            "recip_rank",
            "recall_1000",
            "num_q",
            "num_ret",
            "ndcg_cut_5",
            "ndcg_cut_10",
            "ndcg_cut_15",
            "ndcg_cut_20",
            "ndcg_cut_30",
            "ndcg_cut_100",
            "ndcg_cut_200",
            "ndcg_cut_500",
            "ndcg_cut_1000",
            "map",
            "P_10",
            "judged_10",
            "recall_5",
            "recall_10",
            "recall_15",
            "recall_20",
            "recall_30",
            "recall_100",
            "recall_200",
            "recall_500",
            "recall_1000",
        ]:
            # the space is to avoid getting metrics such as ndcg_cut_100 instead of ndcg_cut_10 as but start with ndcg_cut_10
            if line.startswith(metric + " ") or line.startswith(metric + "\t"):
                metrics[metric] = float(line.split("\t")[-1])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", default="msmarco")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--relevance_threshold", default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--remove_unjudged", action="store_true")
    parser.add_argument("--use_rr", action="store_true", default=False)
    args = parser.parse_args()

    qrels_file = load_qrel_file(args.dataset, args.qrels)
    
    run_file = args.run
    if args.run.lower() == "bm25":
        run = TRECRun(args.dataset)
        run_file = run.run_file


    # filter qrels file by the queries in the run
    run_file, qrels_file = filter_by_run(run_file, qrels_file)

    results = run_trec_eval(run_file, qrels_file, args.relevance_threshold, args.remove_unjudged, args.use_rr)
    if args.json:
        print(json.dumps(results))
    else:
        for (metric, value) in sorted(results.items()):
            print(f"{metric}: {value}")