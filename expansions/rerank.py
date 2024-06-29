import os
import csv
import torch
import sys
import argparse
import pandas as pd
import numpy as np
from math import ceil, exp
import subprocess
from typing import List
import time
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map 
import multiprocessing as mp
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
)
from . import utils
from .dataset import load_corpus, load_queries
from collections import OrderedDict

from .tart.modeling_enc_t5 import EncT5ForSequenceClassification
from .tart.tokenization_enc_t5 import EncT5Tokenizer

# add ColBERT repo to path from `ColBERT`, clone it if you need it
try:
    sys.path.append('ColBERT')
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.modeling.colbert import colbert_score
except Exception as e:
    print(f"Not loading ColBERT...")

try:
    # add splade repo
    sys.path.append('splade')
    from splade.models.transformer_rep import Splade
except Exception as e:
    print(f"Not loading Splade...")


try:
    from huggingface_hub import login
    login(token=os.environ["HUGGINGFACE_TOKEN"])
except Exception as e:
    pass

# Based on https://github.com/castorini/pygaggle/blob/f54ae53d6183c1b66444fa5a0542301e0d1090f5/pygaggle/rerank/base.py#L63
prediction_tokens = {
    'castorini/monot5-small-msmarco-10k':   ['▁false', '▁true'],
    'castorini/monot5-small-msmarco-100k':  ['▁false', '▁true'],
    'castorini/monot5-base-msmarco':        ['▁false', '▁true'],
    'castorini/monot5-base-msmarco-10k':    ['▁false', '▁true'],
    'castorini/monot5-large-msmarco':       ['▁false', '▁true'],
    'castorini/monot5-large-msmarco-10k':   ['▁false', '▁true'],
    'castorini/monot5-base-med-msmarco':    ['▁false', '▁true'],
    'castorini/monot5-3b-med-msmarco':      ['▁false', '▁true'],
    'castorini/monot5-3b-msmarco-10k':      ['▁false', '▁true'],
    'castorini/monot5-3b-msmarco':          ['▁false', '▁true'],
    'unicamp-dl/mt5-base-en-msmarco':       ['▁no'   , '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v2':        ['▁no'   , '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v1':        ['▁no'   , '▁yes'],
}

def clipped(text: str):
    if len(text.split(" ")) > 300:
        return " ".join(text.split(" ")[:300])
    return text


def do_indexing(qid):
    TEMP_DIR_BASE = "temp_bm25_local/"

    index_command = f"""python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input {TEMP_DIR_BASE}/{qid}/docs \
        --index {TEMP_DIR_BASE}/{qid}/indexes \
        --generator DefaultLuceneDocumentGenerator \
        --threads 1"""
    
    # do the commands and verify the files exist
    print("Launching indexing command for qid", qid)
    cmd = [item for item in index_command.replace("\\", "").split(" ") if item.strip() != ""]
    print(cmd)
    subprocess.Popen(cmd)
    

def do_searching(qid):
    TEMP_DIR_BASE = "temp_bm25_local/"
    search_command = f"""python -m pyserini.search.lucene \
        --index {TEMP_DIR_BASE}/{qid}/indexes \
        --topics {TEMP_DIR_BASE}/{qid}/queries.tsv \
        --output {TEMP_DIR_BASE}/{qid}/results.trec \
        --bm25 \
        --hits 1000 \
        --output-format trec"""

    print("Starting search command for query", qid)
    cmd = [item for item in search_command.replace("\\", "").split(" ") if item.strip() != ""]
    print(cmd)
    subprocess.Popen(cmd)

class Reranker:
    def __init__(self, silent=False, batch_size=8, fp16=False, torchscript=False, device=None, custom_prompt=None):
        self.silent = silent
        self.batch_size = batch_size
        self.fp16 = fp16
        self.torchscript = torchscript
        self.custom_prompt = custom_prompt
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        if "bm25" == model_name_or_path:
            return BM25Reranker(**kwargs)
        
        if "tart" in model_name_or_path:
            return TartReranker(model_name_or_path, **kwargs)

        if "splade" in model_name_or_path:
            return SpladeReranker(model_name_or_path, **kwargs)
        
        if "colbert" in model_name_or_path and "llama" not in model_name_or_path:
            return ColBERTReranker(model_name_or_path, **kwargs)
        
        if "castorini" not in model_name_or_path:
            config = AutoConfig.from_pretrained(model_name_or_path)
        else:
            config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

        print(config.architectures)
        seq2seq = any(
            True
            for architecture in config.architectures
            if 'ForConditionalGeneration' in architecture
        )
        if seq2seq:
            if 'flan' in model_name_or_path:
                return FLANT5Reranker(model_name_or_path, **kwargs)
            return MonoT5Reranker(model_name_or_path, **kwargs)
        llama_based = any(
            True
            for architecture in config.architectures
            if 'ForCausalLM' in architecture or "llama" in architecture or "Llama" in architecture
        )
        print(f"LLAMA based is {llama_based}")
        if llama_based:
            is_classification = any(
                True
                for architecture in config.architectures
                if 'ForSequenceClassification' in architecture
            ) or "castorini" in model_name_or_path
            if is_classification:
                return LlamaReranker(model_name_or_path, is_classification=True, **kwargs)
            else:
                return LlamaReranker(model_name_or_path, **kwargs)
        
        if "bert" not in model_name_or_path and "cross-encoder" not in model_name_or_path:
            print(f"Using dense for {model_name_or_path}")
            return DenseReranker(model_name_or_path, **kwargs)
        
        print("Using MonoBERT")
        return MonoBERTReranker(model_name_or_path, **kwargs)


class MonoT5Reranker(Reranker):
    name: str = 'MonoT5'
    prompt_template: str = "Query: {query} Document: {text} Relevant:"

    def __init__(
        self,
        model_name_or_path='castorini/monot5-base-msmarco-10k',
        token_false=None,
        token_true=True,
        torch_compile=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        # if self.fp16:
        #     model_args["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_args)
        self.torch_compile = torch_compile
        if torch_compile:
            self.model = torch.compile(self.model)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(
            model_name_or_path, self.tokenizer, token_false, token_true,
        )
        print(f"Max tokens is {self.tokenizer.model_max_length}")

    def get_prediction_tokens(self, model_name_or_path, tokenizer, token_false=None, token_true=None):
        if not (token_false and token_true):
            if model_name_or_path in prediction_tokens:
                token_false, token_true = prediction_tokens[model_name_or_path]
                token_false_id = tokenizer.get_vocab()[token_false]
                token_true_id  = tokenizer.get_vocab()[token_true]
                return token_false_id, token_true_id
            else:
                print("Loading from base....")
                # raise Exception(f"We don't know the indexes for the non-relevant/relevant tokens for\
                #         the checkpoint {model_name_or_path} and you did not provide any.")
                returned = self.get_prediction_tokens('castorini/monot5-base-msmarco', self.tokenizer)
                print(returned)
                return returned
        else:
            token_false_id = tokenizer.get_vocab()[token_false]
            token_true_id  = tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id

    @torch.inference_mode()
    def rescore(self, pairs: List[List[str]]):
        scores = []
        max_count = 0
        for batch in tqdm(
            utils.chunks(pairs, self.batch_size),
            disable=self.silent,
            desc="Rescoring",
            total=ceil(len(pairs) / self.batch_size),
            bar_format='{l_bar}{bar}{r_bar}\n'
        ):
            prompts = [
                self.prompt_template.format(query=query, text=text)
                for (query, text) in batch
            ]
            tokens = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
                pad_to_multiple_of=(8 if self.torch_compile else None),
            ).to(self.device)
            if tokens.input_ids.shape[1] == self.tokenizer.model_max_length:
                max_count += 1
            # print(tokens["input_ids"].shape)
            output = self.model.generate(
                **tokens,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            batch_scores = output.scores[0]
            batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores += batch_scores[:, 1].exp().tolist()
        print(f"Max count is {max_count}/{len(list(utils.chunks(pairs, self.batch_size)))}")
        return scores


class FLANT5Reranker(MonoT5Reranker):
    name: str = 'FLAN-T5'
    prompt_template: str = """Is the following passage relevant to the query?
Query: {query}
Passage: {text}"""

    def get_prediction_tokens(self, *args, **kwargs):
        yes_token_id, *_ = self.tokenizer.encode('yes')
        no_token_id, *_ = self.tokenizer.encode('no')
        return no_token_id, yes_token_id

class TartReranker(Reranker):
    name: str = 'TART'

    def __init__(
        self,
        model_name_or_path='facebook/tart-full-flan-t5-xl',
        torch_compile=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.instruction = "Retrieve a passage that is relevant to the query"
        self.model = EncT5ForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer =  EncT5Tokenizer.from_pretrained(model_name_or_path)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp16:
            model_args["torch_dtype"] = torch.bfloat16
        # self.torch_compile = torch_compile
        # if torch_compile:
        #     self.model = torch.compile(self.model)
        self.model.to(self.device)
                                             
    @torch.inference_mode()
    def rescore(self, pairs: List[List[str]]):
        scores = []
        for batch in tqdm(
            utils.chunks(pairs, self.batch_size),
            disable=self.silent,
            desc="Rescoring",
            total=ceil(len(pairs) / self.batch_size),
            bar_format='{l_bar}{bar}{r_bar}\n'
        ):
            prompts = ['{0} [SEP] {1}'.format(self.instruction, query) for (query, text) in batch]
            tokens = self.tokenizer(
                prompts,
                [text for (query, text) in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
                pad_to_multiple_of=None,
            ).to(self.device)
            batch_scores = self.model(**tokens).logits
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores += batch_scores[:, 1].exp().tolist()
        return scores
    

class LlamaReranker(Reranker):
    name: str = 'LLAMA-Based'



    def __init__(
        self,
        model_name_or_path: str,
        is_classification: bool = False,
        **kwargs
    ):
        if "torch_compile" in kwargs:
            del kwargs["torch_compile"]
        super().__init__(**kwargs)


        if "chat" in model_name_or_path:
            self.template = LLAMA_CHAT_TEMPLATE = """<s>[INST] <<SYS>>
You are an expert at finding information. Determine if the following document is relevant to the query (true/false).
<</SYS>>Query: {query}
Document: {text}
Relevant: [/INST]"""
        elif "inst" in model_name_or_path: # aka no instruction
            self.template = """Query: {query}
Document: {text}
Relevant: """
        else:
            self.template = """Determine if the following document is relevant to the query (true/false).

Query: {query}
Document: {text}
Relevant: """


        self.is_classification = is_classification
        self.tokenizer = None

        print(self.template)
        # model_name_of_path="google/flan-t5-xl"
        print(model_name_or_path, is_classification)
        # self.model = AutoModelForCausalLM.from_pretrained("/home/person/nfs/decodIR/llama_ir_small/")
        # self.tokenizer = AutoTokenizer.from_pretrained("/home/person/nfs/decodIR/llama_ir_small/", padding_side="left")
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if is_classification:
            if "castorini" in model_name_or_path:
                from peft import PeftModel, PeftConfig
                def get_model(peft_model_name):
                    config = PeftConfig.from_pretrained(peft_model_name)
                    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
                    if base_model.config.pad_token_id is None:
                        base_model.config.pad_token_id = base_model.config.eos_token_id
                    hf_model = PeftModel.from_pretrained(base_model, model_name_or_path)
                    hf_model = hf_model.merge_and_unload()
                    hf_model.eval()
                    return hf_model

                # Load the tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
                self.model = get_model('castorini/rankllama-v1-7b-lora-passage')
                self.model.model_max_length = 1024
                self.model.max_len = 1024
                self.model.config.max_len = 1024
                print("Castorini")
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
                self.model.to(self.device)
        else:
            if "8bit" in model_name_or_path:
                print("Loading in 8bit")
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True, torch_dtype=torch.bfloat16, device_map='auto')
            if "4bit" in model_name_or_path:
                print("Loading in 4bit")
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_4bit=True, torch_dtype=torch.bfloat16, device_map='auto')
            else:
                print("Loading in 16bit")
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
                self.model.to(self.device)

            # self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
        try:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        except Exception as e:
            print(e)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", use_fast=False)



        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
# self.torch_compile = torch_compile
        # if torch_compile:
        # self.model = torch.compile(self.model)
        self.token_false_id = self.tokenizer.get_vocab()["false"]
        self.token_true_id  = self.tokenizer.get_vocab()["true"]
        self.model_name_or_path = model_name_or_path
                                             
    @torch.inference_mode()
    def rescore(self, pairs: List[List[str]]):
        scores = []
        for batch in tqdm(
            utils.chunks(pairs, 8),
            disable=self.silent,
            desc="Rescoring",
            total=ceil(len(pairs) / 8),
            bar_format='{l_bar}{bar}{r_bar}\n'
        ):
            prompts = [self.template.format(query=query, text=text) for (query, text) in batch]
            if "castorini" in self.model_name_or_path:
                texts = [str(text) if text != None else "na" for (_, text) in batch]
                queries = [str(query) if query != None else "na"  for (query, _) in batch]
                print(len(texts), set([type(text) for text in texts]))
                # print(queries[:2])
                encoded_passages = self.tokenizer.encode(texts,
                    add_special_tokens=False,
                    max_length=800,
                    truncation=True
                )
                encoded_queries = self.tokenizer.encode(queries,
                    add_special_tokens=False,
                    max_length=200,
                    truncation=True
                )
                tokens = []
                for query, text in zip(encoded_queries, encoded_passages):
                    cur_tokens = self.tok.prepare_for_model(
                        [self.tok.bos_token_id] + query,
                        [self.tok.bos_token_id] + text,
                        max_length=self.max_q_len + self.max_p_len,
                        truncation='only_first',
                        padding=True,
                        return_token_type_ids=False,
                    )
                    tokens.append(cur_tokens)
            else:
                tokens = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=min(self.tokenizer.model_max_length, 1024),
                    pad_to_multiple_of=None,
                ).to(self.device)
            if "token_type_ids" in tokens:
                del tokens["token_type_ids"]
            if not self.is_classification:
                batch_scores = self.model(**tokens).logits[:, -1, :]
                true_vector = batch_scores[:, self.token_true_id]
                false_vector = batch_scores[:, self.token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores += batch_scores[:, 1].exp().tolist()
            else:
                if "castorini" in self.model_name_or_path:
                    batch_scores = self.model(**tokens).logits
                    print(batch_scores.shape)
                    scores += batch_scores[0]
                else:
                    batch_scores = self.model(**tokens).logits
                    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                    scores += batch_scores[:, 1].exp().tolist()

        return scores


class MonoBERTReranker(Reranker):
    name: str = 'MonoBERT'

    def __init__(
        self,
        model_name_or_path='cross-encoder/ms-marco-MiniLM-L-6-v2',
        torch_compile=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        # model_name_or_path = "castorini/monobert-large-msmarco"
        if self.fp16:
            model_args["torch_dtype"] = torch.float16
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            **model_args,
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    @torch.inference_mode()
    def rescore(self, pairs: List[List[str]]):
        scores = []
        for batch in tqdm(
            utils.chunks(pairs, self.batch_size),
            disable=self.silent,
            desc="Rescoring",
            total=ceil(len(pairs) / self.batch_size),
            bar_format='{l_bar}{bar}{r_bar}\n'
        ):
            queries, texts = zip(*batch)
            try:
                tokens = self.tokenizer(
                    queries,
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.tokenizer.model_max_length,
                ).to(self.device)
            except Exception as e:
                print(e, queries, texts)
                raise e
            output = self.model(**tokens).logits
            if output[0].shape[-1] > 1: # distribution over tokens
                batch_scores = torch.nn.functional.log_softmax(output, dim=1)
                scores += batch_scores[:, 1].exp().tolist()
            else:
                scores += torch.nn.functional.sigmoid(output).cpu().detach().tolist()

        return scores


class BM25Reranker(Reranker):
    name: str = 'BM25'

    def __init__(
        self,
        model_name_or_path=None,
        torch_compile: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.temp_location = "temp_bm25_local/"
   
    def reset_directory(self):
        subprocess.Popen(["rm", "-rf", self.temp_location])
        time.sleep(120)

    @torch.inference_mode()
    def rescore(self, pairs: List[List[str]]):
        self.reset_directory()
        # rearrange by query and docs
        scores = []
        all_groups = []
        cur_group = []
        cur_query = pairs[0][0]
        for idx, (query, text) in enumerate(pairs):
            if cur_query == query:
                cur_group.append(text)
            else:
                all_groups.append((cur_query, cur_group))
                cur_group = [text]
                cur_query = query
        all_groups.append((pairs[-1][0], cur_group))

        # do creation here to avoid lock issues
        for qid_num in tqdm(range(len(all_groups))):
            TEMP_DIR_BASE = "temp_bm25_local/"
            os.makedirs(os.path.join(TEMP_DIR_BASE, str(qid_num), "docs"), exist_ok=True)
            os.makedirs(os.path.join(TEMP_DIR_BASE, str(qid_num), "indexes"), exist_ok=True)

            # write out docs and query
            query, texts = all_groups[qid_num]
            docs = pd.DataFrame({"id": range(len(texts)), "contents": texts})
            docs.to_json(f"{TEMP_DIR_BASE}/{qid_num}/docs/docs.json", orient="records", lines=True)
            queries = pd.DataFrame({"query_id": [0], "contents": [query]})
            queries.to_csv(f"{TEMP_DIR_BASE}/{qid_num}/queries.tsv", sep="\t", index=False, header=None)


        # do multiprocessing on this
        print(f"Starting batch indexing")
        for i in range(len(all_groups)):
            do_indexing(i)
        print("Finished indexing commands...")
        time.sleep(120)

        cur_iter = 0
        while not os.path.isfile(f"{self.temp_location}/{len(all_groups) - 1}/indexes/write.lock"):
            print("Waiting for indexing to finish")
            time.sleep(120)
            cur_iter += 1
            if cur_iter > 100:
                raise Exception("Something went wrong with the indexing")

        print("Starting batch searching")
        for i in range(len(all_groups)):
            do_searching(i)
        print("Finished searching commands...")
        time.sleep(120)

        cur_iter = 0
        while not os.path.isfile(f"{self.temp_location}/{len(all_groups) - 1}/results.trec"):
            print("Waiting for searching to finish")
            time.sleep(120)
            cur_iter += 1
            if cur_iter > 100:
                raise Exception("Something went wrong with the searching")

        time.sleep(120)

        # now we can use BM25 to rerank each set
        idx = 0
        for query, texts in tqdm(
            all_groups,
            disable=self.silent,
            desc="Rescoring",
            total=len(all_groups),
            bar_format='{l_bar}{bar}{r_bar}\n'
        ):
            # read in results
            results = pd.read_csv(f"{self.temp_location}/{idx}/results.trec", sep="\s+", header=None, index_col=None)
            results_tuples = list(zip(results[2], results[4])) # id and score
            
            if len(results_tuples) < len(texts):
                unused_docs = set(range(len(texts))) - set(results[2])
                for doc_id in unused_docs:
                    results_tuples.append((doc_id, 0.0))

            # add to scores
            # sort results by doc id
            results_sorted = sorted(results_tuples, key=lambda x: x[0])
            scores += [float(item[-1]) for item in results_sorted]

            idx += 1

        self.reset_directory()
        return scores


class DenseReranker(Reranker):
    name: str = 'DenseReranker'

    def __init__(
        self,
        model_name_or_path='facebook/contriever',
        torch_compile=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from sentence_transformers import SentenceTransformer, util
        if "dpr" in model_name_or_path:
            self.question_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
            self.embedder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
            self.is_dpr = True
            self.question_encoder.max_seq_length = 510
            self.embedder.max_seq_length = 510
        else:
            self.embedder = SentenceTransformer(model_name_or_path)
            self.embedder.max_seq_length = 510

            self.is_dpr = False


    @torch.inference_mode()
    def rescore(self, pairs: List[List[str]]):
        scores = []
        from sentence_transformers import SentenceTransformer, util
        new = True
        all_groups = []
        cur_group = []
        cur_query = pairs[0][0]
        for idx, (query, text) in enumerate(pairs):
            if cur_query == query:
                cur_group.append(text)
            else:
                all_groups.append((cur_query, cur_group))
                cur_group = [text]
                cur_query = query
        all_groups.append((pairs[-1][0], cur_group))
        
        for query, texts in tqdm(
            all_groups,
            disable=self.silent,
            desc="Rescoring",
            total=len(all_groups),
            bar_format='{l_bar}{bar}{r_bar}\n'
        ):
            corpus_embeddings = self.embedder.encode(texts, convert_to_tensor=True)
            corpus_embeddings = corpus_embeddings.to(self.device)
            if not self.is_dpr:
                corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

            if self.is_dpr:
                query_embeddings = self.question_encoder.encode(query, convert_to_tensor=True)
            else:
                query_embeddings = self.embedder.encode([query], convert_to_tensor=True)
                query_embeddings = util.normalize_embeddings(query_embeddings)
            
            query_embeddings = query_embeddings.to(self.device)
            hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score, top_k=len(texts))

            hits_ordered_by_num = sorted(hits[0], key=lambda x: x['corpus_id'])
            scores += [hit['score'] for hit in hits_ordered_by_num]
        return scores


class SpladeReranker(Reranker):
    name: str = 'SpladeReranker'

    def __init__(
        self,
        model_name_or_path="naver/splade-cocondenser-ensembledistil",
        torch_compile=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Splade(model_name_or_path, agg="max")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    @torch.inference_mode()
    def rescore(self, pairs: List[List[str]]):
        scores = []
        new = True
        all_groups = []
        cur_group = []
        cur_query = pairs[0][0]
        for idx, (query, text) in enumerate(pairs):
            if cur_query == query:
                cur_group.append(text)
            else:
                all_groups.append((cur_query, cur_group))
                cur_group = [text]
                cur_query = query
        all_groups.append((pairs[-1][0], cur_group))
        
        for query, texts in tqdm(
            all_groups,
            disable=self.silent,
            desc="Rescoring",
            total=len(all_groups),
            bar_format='{l_bar}{bar}{r_bar}\n'
        ):
            docs_tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            doc_reps = self.model(d_kwargs=docs_tokenized)["d_rep"].squeeze()  
            q_rep = self.model(d_kwargs=self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512))["d_rep"].squeeze()  
            hits = np.dot(doc_reps, q_rep)
            scores += hits.tolist()
        return scores



class ColBERTReranker(Reranker):
    name: str = 'ColBERTReranker'

    def __init__(
        self,
        model_name_or_path="colbertv2",
        torch_compile=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading colbert with {model_name_or_path}")
        self.model = Checkpoint(model_name_or_path, colbert_config=ColBERTConfig(query_maxlen=510, doc_maxlen=510))


    @torch.inference_mode()
    def rescore(self, pairs: List[List[str]]):
        scores = []
        new = True
        all_groups = []
        cur_group = []
        cur_query = pairs[0][0]
        for idx, (query, text) in enumerate(pairs):
            if cur_query == query:
                cur_group.append(text)
            else:
                all_groups.append((cur_query, cur_group))
                cur_group = [text]
                cur_query = query
        all_groups.append((pairs[-1][0], cur_group))
        
        for query, texts in tqdm(
            all_groups,
            disable=self.silent,
            desc="Rescoring",
            total=len(all_groups),
            bar_format='{l_bar}{bar}{r_bar}\n'
        ):
            q_rep = self.model.queryFromText([query], bsize=1)
            doc_reps = self.model.docFromText(texts, bsize=100)[0]

            hits = colbert_score(q_rep, doc_reps, D_mask=torch.ones((doc_reps.shape[0], doc_reps.shape[1])))
            scores += hits.tolist()
        return scores


def split_dict_into_chunks(input_dict, num_chunks, keep_idx):
    chunk_size = len(input_dict) // num_chunks
    sorted_input = [(k, input_dict[k]) for k in sorted(input_dict.keys())]
    ordered_dict = OrderedDict(sorted_input)
    input_list = list(ordered_dict.items())

    start = chunk_size * keep_idx
    end = start + chunk_size if keep_idx < num_chunks - 1 else None
    return dict(input_list[start:end])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='castorini/monot5-small-msmarco-100k',
            type=str, required=False, help="Reranker model.")
    parser.add_argument("--input_run", default=None, type=str,
                        help="Initial run to be reranked.")
    parser.add_argument("--output_run", default=None, type=str, required=True,
                        help="Path to save the reranked run.")
    parser.add_argument("--dataset", default=None, type=str,
                        help="Dataset name from BEIR collection.")
    parser.add_argument('--dataset_source', default='ir_datasets',
                        help="The dataset source: ir_datasets or pyserini")
    parser.add_argument("--corpus", default=None, type=str,
                        help="Document collection `doc_id` and `text` fields in CSV format.")
    parser.add_argument("--queries", default=None, type=str,
                        help="Queries collection with `query_id` and `text` fields in CSV format.")
    parser.add_argument("--device", default=None, type=str,
                        help="CPU or CUDA device.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use FP16 weights during inference.")
    parser.add_argument("--torch_compile", action="store_true",
                        help="Whether to compile the model with `torch.compile`.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for inference.")
    parser.add_argument("--top_k", default=1_000, type=int,
                        help="Top-k documents to be reranked for each query.")
    parser.add_argument("--chunk_queries", default=1, type=int,
                        help="Split the queries into n chunks (for parallel scoring).")
    parser.add_argument("--chunk_idx", default=0, type=int,
                        help="The chunk index of the queries to rerank in this process.")
    parser.add_argument("--corpus_combination", default=None, type=str, help="how to combine new corpus if used with args.dataset_name")
    parser.add_argument("--queries_combination", default=None, type=str, help="how to combine new queries if used with args.dataset_name")
    parser.add_argument("--max_num_queries", default=None, type=int, help="how many queries to use from the dataset")
    parser.add_argument("--max_num_docs", default=None, type=int, help="how many docs to use from the dataset")
    parser.add_argument("--custom_prompt", default="none", type=str, help="custom prompt to use for models")
    args = parser.parse_args()

    if args.max_num_docs == -1:
        args.max_num_docs = None

    if args.dataset:           
        corpus = load_corpus(args.dataset, source=args.dataset_source)
        corpus = dict(zip(corpus['doc_id'], corpus['text']))
        print(f"Loaded {len(corpus)} documents from {args.dataset}.")
        skip_alt = args.corpus_combination in [None, "", "none"]
        if args.corpus not in ["none", None] and not skip_alt:
            print(f"Combining with {args.corpus}")
            alt_corpus = pd.read_json(args.corpus, lines=True)
            id_col = alt_corpus.columns[:1]
            alt_corpus[id_col] = alt_corpus[id_col].astype(str)
            alt_corpus = alt_corpus.set_index(id_col.tolist())
            alt_corpus = alt_corpus["text"].to_dict()
            if args.corpus_combination == "replace":
                corpus = alt_corpus
            elif args.corpus_combination == "append":
                for k, v in alt_corpus.items():
                    corpus[k] = corpus[k] + " " + v
            elif args.corpus_combination == "prepend":
                for k, v in alt_corpus.items():
                    corpus[k] = v + " " + corpus[k]
            else:
                raise NotImplementedError(f"Unknown corpus combination method: {args.corpus_combination}")
        queries = load_queries(args.dataset, source=args.dataset_source)
        print(f"Original number of queries: {len(queries)}")
        if args.queries not in ["none", None]:
            alt_queries_df = pd.read_json(args.queries, lines=True)
            alt_queries = dict(zip(alt_queries_df["_id"].astype(str), alt_queries_df["text"]))
            assert set(alt_queries.keys()) - set(queries.keys()) == set(), "queries in alt_queries not in queries"
            if args.queries_combination == "replace":
                queries = alt_queries
            elif args.queries_combination == "append":
                for key, value in queries.items():
                    if key in alt_queries:
                        queries[key] = queries[key] + " " + alt_queries[key]
            elif args.queries_combination == "append5x":
                for key, value in queries.items():
                    if key in alt_queries:
                        queries[key] = " ".join([queries[key]] * 5 + [alt_queries[key]])            
            elif args.queries_combination == "prepend":
                for key, value in queries.items():
                    if key in alt_queries:
                        queries[key] = alt_queries[key] + " " + queries[key]
            else:
                raise NotImplementedError(f"Unknown queries combination method: {args.queries_combination}")
    else:
        if '.csv' in args.corpus:
            corpus = pd.read_csv(args.corpus, index_col=0)
            corpus.index = corpus.index.astype(str)
            corpus = corpus.iloc[:, 0].to_dict()
        elif '.json' in args.corpus:
            corpus = pd.read_json(args.corpus, lines=True)
            id_col, text_col = corpus.columns[:2]
            corpus[id_col] = corpus[id_col].astype(str)
            corpus = corpus.set_index(id_col)
            corpus = corpus[text_col].to_dict()

        if '.csv' in args.queries:
            queries = pd.read_csv(args.queries, index_col=0)
            queries.index = queries.index.astype(str)
            queries = queries.iloc[:, 0].to_dict()
        elif '.tsv' in args.queries:
            queries = pd.read_csv(args.queries, header=None, sep='\t', index_col=0)
            queries.index = queries.index.astype(str)
            queries = queries.iloc[:, 0].to_dict()


    input_run = args.input_run
    if args.dataset and not args.input_run:
        input_run = args.dataset


    model = Reranker.from_pretrained(
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        fp16=args.fp16,
        device=args.device,
        torch_compile=args.torch_compile,
        custom_prompt=args.custom_prompt,
        # torchscript=args.torchscript,
    )

    print(input_run, args.max_num_queries, args.max_num_docs)
    run = utils.TRECRun(input_run, num_queries=args.max_num_queries, num_docs=args.max_num_docs)
    print(f"Loaded {len(run.df.qid.unique())} queries from {input_run}.")
    queries = {k: v for k, v in queries.items() if k in run.df.qid.unique()}
    if args.chunk_queries > 1:
        queries = split_dict_into_chunks(queries, args.chunk_queries, args.chunk_idx)
        
    print(f"Reranking {len(queries)} queries")
    print(f"Reranking {len(corpus)} docs")

    save_local = os.environ.get("SAVE_LOCAL", False)
    if save_local:
        # make it unique quries and docs only
        docs = run.df[run.df.docid.isin(run.df["docid"].unique().tolist())]
        queries = run.df[run.df.qid.isin(run.df["qid"].unique().tolist())]

        docs["_id"] = docs["docid"]
        docs["text"] = docs["docid"].apply(lambda x: corpus[x])
        docs[["_id", "text"]].to_json("llm-based-expansions-eval-datasets/msmarco_dev/corpus.jsonl", orient="records", lines=True)

        queries["_id"] = queries["qid"]
        queries["text"] = queries["qid"].apply(lambda x: queries[x])
        queries[["_id", "text"]].to_json("llm-based-expansions-eval-datasets/msmarco_dev/queries.jsonl", orient="records", lines=True)

        breakpoint()
    # for doc stuff here, try re-ranking
    # average_len = np.mean([len(corpus[x].split(" ")) for x in corpus.keys()])
    run.rerank(model, queries, corpus, top_k=args.top_k)  
    
    if args.chunk_queries > 1:
        print(f"Saving run to {args.output_run}.{args.chunk_idx}.{args.chunk_queries}")
        run.save(args.output_run + f'.{args.chunk_idx}.{args.chunk_queries}')
    else:
        print(f"Saving run to {args.output_run}, because chunk_queries={args.chunk_queries}")
        run.save(args.output_run)
