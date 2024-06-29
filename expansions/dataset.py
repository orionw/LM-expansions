import ftfy
import json
import os
import random
import pandas as pd
from tqdm.auto import tqdm
import argparse
from .utils import TRECRun, BEIR_DATASETS
import ir_datasets

DATASET_DOC_FORMAT = {
    "medline/2017/trec-pm-2017": "title,abstract",
    "clinicaltrials/2021/trec-ct-2022": "title,condition,summary,detailed_description,eligibility",
    "clinicaltrials/2021/trec-ct-2021": "title,condition,summary,detailed_description,eligibility",
}

DATASET_QUERY_FORMAT = {
    "medline/2017/trec-pm-2017": "disease,gene,demographic",
    "clinicaltrials/2021/trec-ct-2022": "text",
    "clinicaltrials/2021/trec-ct-2021": "text",
}

LOCAL_DATASETS = [
    "msmarco_dev",
    "scifact_refute",
    "tot",
    "wikiqa",
]

PATH_TO_LOCAL = "./llm-based-expansions-eval-datasets"

def get_query_text(query, text_type, dataset):
    if text_type == "standard":
        if dataset is None:
            return query["text"]
        else:
            return query.text
    else:
        attributes = [getattr(query, field) for field in text_type.split(",") if field in dataset.queries_cls()._fields]
        attributes = [item for item in attributes if item not in ["", None]]
        attributes = [item[0] if type(item) == list else item for item in attributes]
        return " ".join(attributes)


def get_doc_text(doc, text_type, dataset):
    if text_type == "standard":
        try:
            return f"{doc.title} {doc.text}" if "title" in dataset.docs_cls()._fields else doc.text
        except:
            return f'{doc["title"]} {doc["text"]}' if "title" in doc else doc["text"]
    else:
        attributes = [getattr(doc, field) for field in text_type.split(",") if field in dataset.docs_cls()._fields]
        attributes = [item for item in attributes if item not in ["", None]]
        attributes = [item[0] if type(item) == list else item for item in attributes]
        return " ".join(attributes)

def load_corpus(dataset_name, source='ir_datasets'):
    texts = []
    docs_ids = []

    text_type = DATASET_DOC_FORMAT[dataset_name] if dataset_name in DATASET_DOC_FORMAT else "standard"
    if dataset_name in LOCAL_DATASETS:
        source = 'local'

    if source == 'ir_datasets' and dataset_name not in BEIR_DATASETS:
        import ir_datasets

        identifier = f'beir/{dataset_name}'
        if identifier in ir_datasets.registry._registered:
            dataset = ir_datasets.load(identifier)
        else:
            dataset = ir_datasets.load(dataset_name)

        for doc in tqdm(
            dataset.docs_iter(), total=dataset.docs_count(), desc="Loading documents from ir-datasets"
        ):
            texts.append(
                ftfy.fix_text(
                    get_doc_text(doc, text_type, dataset)
                )
            )
            docs_ids.append(doc.doc_id)
    elif source == 'local':
        docs_ids = []
        texts = []
        with open(f"{PATH_TO_LOCAL}/{dataset_name}/corpus.jsonl", "r") as fin:
            for line in tqdm(fin):
                doc = json.loads(line)
                texts.append(
                    ftfy.fix_text(
                        get_doc_text(doc, text_type, None)
                    )
                )
                docs_ids.append(doc['_id'])
    else:
        from pyserini.search.lucene import LuceneSearcher
        from pyserini.prebuilt_index_info import TF_INDEX_INFO

        identifier = f'beir-v1.0.0-{dataset_name}.flat'
        if dataset_name == "msmarco":
            print("Loading MSMARCO passage index")
            dataset = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage-full")
            id_key = "id"
            text_key = "contents"
        elif identifier in TF_INDEX_INFO:
            id_key = "_id"
            text_key = "text"
            dataset = LuceneSearcher.from_prebuilt_index(identifier)
        else:
            id_key = "_id"
            text_key = "text"
            dataset = LuceneSearcher.from_prebuilt_index(dataset_name)
        
        for idx in tqdm(range(dataset.num_docs), desc="Loading documents from Pyserini"):
            doc = json.loads(dataset.doc(idx).raw())
            if text_key == "text":
                texts.append(
                    ftfy.fix_text(
                        get_doc_text(doc, text_type, dataset)
                    )
                )
            else:
                texts.append(
                    ftfy.fix_text(
                        doc[text_key]
                    )
                )
            docs_ids.append(doc[id_key])

    df = pd.DataFrame({'doc_id': docs_ids, 'text': texts})
    # assert len(df) == len(set(df.doc_id)), f"Found {len(df)} documents, but {len(set(df.doc_id))} unique doc_ids"
    return df


def load_queries(dataset_name, source='ir_datasets'):
    queries = {}
    text_type = DATASET_QUERY_FORMAT[dataset_name] if dataset_name in DATASET_QUERY_FORMAT else "standard"
    if dataset_name in LOCAL_DATASETS:
        source = 'local'


    if source == 'ir_datasets' and dataset_name not in BEIR_DATASETS:
        import ir_datasets
        try:
            dataset = ir_datasets.load(f'beir/{dataset_name}')
        except Exception as e:
            dataset = ir_datasets.load(dataset_name)

        for query in dataset.queries_iter():
            qtext = get_query_text(query, text_type, dataset)
            queries[query.query_id] = ftfy.fix_text(qtext)
    elif source == 'local':
        with open(f"{PATH_TO_LOCAL}/{dataset_name}/queries.jsonl", "r") as fin:
            for line in fin:
                query = json.loads(line)
                qtext = get_query_text(query, text_type, None)
                queries[query['_id']] = ftfy.fix_text(qtext)
    else:
        from pyserini.search import get_topics

        identifier = f'beir-v1.0.0-{dataset_name}-test'
        if dataset_name == "msmarco":
            identifier = "msmarco-passage-dev-subset"
        for (qid, data) in get_topics(identifier).items():
            queries[str(qid)] = ftfy.fix_text(data["query"] if "query" in data else data["title"])  # assume 'title' is the query

    return queries

def load_queries_in_run(dataset_name, num_queries: int = None):
    run = TRECRun(dataset_name, num_queries=num_queries)
    return run.df.qid.unique().tolist()


def load_docs_in_run(dataset_name: str, num_queries: int = None, num_docs: int = None):
    run = TRECRun(dataset_name, num_queries=num_queries, num_docs=num_docs)
    return run.df.docid.unique().tolist()


def is_valid_qrel(dataset_name, relevance):
    return True
    if dataset_name == "clinicaltrials/2021/trec-ct-2021":
        if relevance == 2:
            return True
        else:
            return False
    else:
        return True


def make_qrels_file(dataset_name: str) -> str:
    dataset = ir_datasets.load(dataset_name)
    qrels_location = ir_datasets.util.home_path() / 'tmp_qrels.txt'
    with open(qrels_location, "w") as f:
        doc_ids = set()
        for qrel in dataset.qrels_iter():
            if qrel.doc_id not in doc_ids and is_valid_qrel(dataset_name, qrel.relevance): # sometimes there are dups?
                doc_ids.add(qrel.doc_id)
                # qrel # namedtuple<query_id, doc_id, relevance, iteration>
                f.write(f'{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n')
    return qrels_location