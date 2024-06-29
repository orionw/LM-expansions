import ast
import json
import pandas as pd
from nltk.tokenize import sent_tokenize
from .dataset import load_corpus, load_queries, load_docs_in_run, load_queries_in_run
from .utils import TRECRun

### utils ###
flatten = lambda l: [item for sublist in l for item in sublist]


def remove_num(string: str) -> str:
  string = string.replace("\n", "")
  if not len(string):
      return string
  
  if "- " in string[:2]:
    string = string[2:]
  if len(string) and string[0] == '"' and string[-1] == '"':
    string = string[1:-1]
  for i in range(20):
    if f"{i}. " in string[:len(f"{i}. ")]:
      string = string[len(f"{i}. "):]
      return string
    if f"{i}: " in string[:len(f"{i}: ")]:
      string = string[len(f"{i}: "):]
      return string
  return string


### Preprocess functions ###

def jsonl_text(name: str, **kwargs) -> list:
    lines = []
    with open(name, "r") as f:
        for line in f:
            lines.append(json.loads(line)["text"])
    return lines


def jsonl_title_and_text_combined(name: str, **kwargs) -> list:
    lines = []
    with open(name, "r") as f:
        for line in f:
            lines.append(json.loads(line)["title"] + " " + json.loads(line)["text"])
    return lines


def jsonl_title_and_text_separate(name: str, **kwargs) -> list:
    lines = []
    with open(name, "r") as f:
        for line in f:
            loaded_line = json.loads(line)
            lines.append((loaded_line["title"], loaded_line["text"]))
    return lines

def jsonl_title_and_text_newline(name: str, **kwargs) -> list:
    lines = []
    with open(name, "r") as f:
        for line in f:
            loaded_line = json.loads(line)
            lines.append(loaded_line["title"] + "\n" + loaded_line["text"])
    return lines


def from_corpus(name: str, **kwargs) -> list:
    return load_corpus(name).to_dict("records")

def from_query(name: str, **kwargs) -> list:
    return load_queries(name)

def from_query_and_doc(name: str, **kwargs) -> list:
    return (load_queries(name), load_corpus(name).to_dict("records"))

def from_list(data: list, **kwargs) -> list:
    return [item["text"] for item in data], [item["doc_id"] for item in data]

def filter_docs_by_run(input, dataset_name: str, num_docs=100, **kwargs) -> list:
    docs = set(load_docs_in_run(dataset_name, num_docs=num_docs))
    subset_docs = [item for item in input if item["doc_id"] in docs]
    print(f"Filtered {len(input)} docs to {len(subset_docs)} docs")
    print(f"Example : {subset_docs[0]}")
    return subset_docs

def filter_docs_by_run_300(input, dataset_name: str, **kwargs) -> list:
    return filter_docs_by_run(input, dataset_name, 300, **kwargs)

def filter_docs_by_run_1k(input, dataset_name: str, **kwargs) -> list:
    return filter_docs_by_run(input, dataset_name, 1000, **kwargs)

def filter_docs_by_run_10k(input, dataset_name: str, **kwargs) -> list:
    return filter_docs_by_run(input, dataset_name, 10000, **kwargs)


def filter_queries_by_run(input, dataset_name: str, **kwargs) -> list:
    queries = set(load_queries_in_run(dataset_name))
    only_test_queries = []
    ids = []
    for id_key, query_value in input.items():
        if id_key in queries:
            only_test_queries.append(query_value)
            ids.append(id_key)
    print(f"Filtered {len(input)} queries to {len(only_test_queries)} queries")
    print(f"Example : {only_test_queries[0]}")
    return only_test_queries, ids


def filter_queries_by_run_300(input, dataset_name: str, **kwargs) -> list:
    queries = set(load_queries_in_run(dataset_name, 300))
    only_test_queries = []
    ids = []
    for id_key, query_value in input.items():
        if id_key in queries:
            only_test_queries.append(query_value)
            ids.append(id_key)
    print(f"Filtered {len(input)} queries to {len(only_test_queries)} queries")
    print(f"Example : {only_test_queries[0]}")
    return only_test_queries, ids

  
def filter_queries_and_docs_by_run(input, dataset_name: str, num_queries = 9999999, num_docs = 100, **kwargs) -> list:
    queries_input, docs_input = input
    print(f"Using {num_queries} queries and {num_docs} docs")
    docs = set(load_docs_in_run(dataset_name, num_queries=num_queries, num_docs=num_docs))
    subset_docs = [item for item in docs_input if item["doc_id"] in docs]
    print(f"Filtered {len(docs_input)} docs to {len(subset_docs)} docs")
    print(f"Example : {subset_docs[0]}")

    final_docs, final_doc_ids = from_list(subset_docs)
    queries = list(queries_input.values())

    ## need to find the top 5 queries per document using semantic search
    from sentence_transformers import SentenceTransformer, util
    import torch
    from torch.nn import CosineSimilarity
    model = SentenceTransformer('all-mpnet-base-v2')
    # print(f"Queries shape : {len(queries)}")
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    index2query = {i: q for i, q in enumerate(queries)}
    # print(f"Query embeddings shape : {query_embeddings.shape}")
   
    cos = CosineSimilarity(dim=1, eps=1e-6)
  
    top_queries_per_doc = []
    N_QUERIES = 3
    for final_doc in final_docs:
        doc_embeddings = model.encode([final_doc], convert_to_tensor=True)
        # print(f"Doc embeddings shape : {doc_embeddings.shape}")
        query_doc_scores = util.cos_sim(doc_embeddings, query_embeddings)
        # print(query_doc_scores)
        top_qs = torch.topk(torch.tensor(query_doc_scores), N_QUERIES)
        # print(f"Top 5 scores for doc {i} : {top_five}")
        real_qs = [index2query[j] for j in top_qs.indices.tolist()[0]]
        # print(f"Top 5 queries for doc {i} : {real_qs}")
        # add numbering to the top 5, like 1: query, 2: query, etc.
        real_qs = [f"{j+1}: {real_qs[j]}" for j in range(len(real_qs))]
        top_queries_per_doc.append("\n".join(real_qs))

    print(f"Example : {top_queries_per_doc[0]}")
            
    # this is a doc expansion method don't need query ids
    return list(zip(top_queries_per_doc, final_docs)), final_doc_ids


def filter_queries_and_docs_by_run_300_100(input, dataset_name: str, **kwargs) -> list:
    return filter_queries_and_docs_by_run(input, dataset_name, 300, 100, **kwargs)



def filter_queries_and_docs_by_run_prf(input, dataset_name: str, num_queries = 9999999, num_docs = 100, **kwargs) -> list:
    queries_input, docs_input = input
    docs = set(load_docs_in_run(dataset_name, num_queries, num_docs))
    subset_docs = [item for item in docs_input if item["doc_id"] in docs]
    print(f"Filtered {len(docs_input)} docs to {len(subset_docs)} docs")
    print(f"Example : {subset_docs[0]}")

    final_docs, final_doc_ids = from_list(subset_docs)
    queries = list(queries_input.values())
    query_ids = list(queries_input.keys())

    ## need to find the top 5 queries per document using semantic search
    from sentence_transformers import SentenceTransformer, util
    import torch
    from torch.nn import CosineSimilarity
    model = SentenceTransformer('all-mpnet-base-v2')
    # print(f"Queries shape : {len(queries)}")
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    doc_embeddings = model.encode(final_docs, convert_to_tensor=True)
    results = util.semantic_search(query_embeddings, doc_embeddings, top_k=3)

    index2doc = {i: d for i, d in enumerate(final_docs)}
    real_docs = [[index2doc[doc_id["corpus_id"]] for doc_id in result_list] for result_list in results]
    formatted_docs = [[f"Inspiration Document {j+1}: {real_docs[i][j]}" for j in range(len(real_docs[i]))] for i in range(len(real_docs))]
    top_docs_per_query = ["\n\n".join(formatted_docs[i]) for i in range(len(formatted_docs))]

    print(f"Example : {top_docs_per_query[0]}")
            
    # this is a doc expansion method don't need query ids
    return list(zip(queries, top_docs_per_query)), query_ids


def filter_queries_and_docs_by_run_prf_1k(input, dataset_name: str, **kwargs) -> list:
    return filter_queries_and_docs_by_run_prf(input, dataset_name, 99999999999, 1000, **kwargs)

def filter_queries_and_docs_by_run_prf_10k(input, dataset_name: str, **kwargs) -> list:
    return filter_queries_and_docs_by_run_prf(input, dataset_name, 99999999999, 10000, **kwargs)


ID_LIST = {
    "quora": ["63928", "388832"],
    "fiqa": ["5021"],
    # "nfcorpus": ["PLAIN-3271"],
    # "scifact_refute": ["338", "847", "1232"]
    "arguana": ['test-politics-grcrgshwbr-pro04a', 'test-health-dhgsshbesbc-pro02a', 'test-science-wsihwclscaaw-pro03a', 'test-law-hrpepthwuto-con03a', 'test-law-cpilhbishioe-pro01a', 'test-law-ralhrilglv-con01a', 'test-culture-mmctghwbsa-con02a', 'test-culture-cgeeghwmeo-con01a', 'test-philosophy-elhbrd-con02a', 'test-international-gsciidffe-pro04a', 'test-free-speech-debate-nshbbsbfb-pro04a', 'test-international-atiahblit-pro02a', 'test-health-dhgsshbesbc-con03a', 'test-international-aglhrilhb-pro01a', 'test-science-wsihwclscaaw-con02a', 'test-politics-grcrgshwbr-con04a', 'test-politics-ypppdghwid-con02a', 'test-international-ehbfe-con01a', 'test-health-hdond-con03a', 'test-religion-grcrgshwbr-con03a', 'test-law-lghbacpsba-pro05a', 'test-religion-grcrgshwbr-con04a', 'test-law-hrpepthwuto-pro03a', 'test-free-speech-debate-nshbcsbawc-con03a', 'test-international-bldimehbn-pro03a', 'test-politics-oepdlhfcefp-pro02a', 'test-philosophy-pppgshbsd-con05a', 'test-philosophy-eppphwlrtjs-pro05a', 'test-religion-grcrgshwbr-pro04a', 'test-religion-yercfrggms-pro07a', 'test-free-speech-debate-nshbbsbfb-con01a', 'test-sport-tshbmlbscac-pro01a', 'test-international-epvhwhranet-pro03a', 'test-science-eassgbatj-con03a', 'test-culture-mmciahbans-con03a']
}

def filter_queries_and_docs_by_run_prf_by_id(input, dataset_name: str, num_queries = 9999999, num_docs = 100, **kwargs) -> list:
    queries_input, docs_input = input
    # filter queries by those in the ID_LIST
    queries_input = {k: v for k, v in queries_input.items() if str(k) in ID_LIST[dataset_name]}
    assert len(queries_input) < 150, len(queries_input)
    print(f"Filtered {len(queries_input)} queries to {len(queries_input)} queries")
    return filter_queries_and_docs_by_run_prf((queries_input, docs_input), dataset_name, num_queries, num_docs, **kwargs)


### Postprocess functions ###

def newline_colon(outputs: list, **kwargs) -> list:
    all_results = []
    for e_idx, entry in enumerate(outputs):
        cur_output = []
        uses_double_newline = "\n\n" in entry.strip()
        splitting_char = "\n\n" if uses_double_newline else "\n"
        use_next_tag = False
        if uses_double_newline and '"\nComma Separated List' in entry:
            # llama messes this up
            entry = entry.replace("\nComma Separated List", "\n\nComma Separated List")
            entry = entry.replace("Comma Separated List of 10 important New Keywords: \n\n", "Comma Separated List of 10 important New Keywords: \n")
            entry = entry.replace("New Question (combining Input and New Keywords, only **one** new question that expands upon the Input): \n\n", "New Question (combining Input and New Keywords, only **one** new question that expands upon the Input): ")
            entry = entry.replace("Please ensure that the new keywords and question provide additional context and insight into the original input. are creative and provide new context and insight into the topic.", "")
            entry = entry.replace("Please ensure that the new keywords and question provide additional context and insight into the original input.", "")

        for item in entry.split(splitting_char):
            if item.strip() != "":
                item_output = item.split(": ")[-1]
                tag = item.split(": ")[0]
                inst = {"text": remove_num(item_output.strip()), "tag": tag}
                if inst["text"] in [None, ""]:
                    if use_next_tag:
                        inst["tag"] = prev_tag
                        use_next_tag = False
                        prev_tag = None
                    else:
                        use_next_tag = True
                        prev_tag = tag
                
                if not use_next_tag:
                    cur_output.append(inst)

        all_results.append(cur_output)
    return all_results


def newlines_and_colon(outputs: list, **kwargs) -> list:
    all_results = []
    for e_idx, entry in enumerate(outputs):
        cur_output = []
        uses_double_newline = "\n\n" in entry.strip()
        splitting_char = "\n\n" if uses_double_newline else "\n"
        for item in entry.split(splitting_char):
            if item.strip() != "":
                if ":" in item:
                    item_output = item.split(": ")[-1]
                    tag = item.split(": ")[0]
                    inst = {"text": remove_num(item_output.strip()), "tag": tag}
                    # if inst["text"] in [None, ""]:
                    #     breakpoint()
                    cur_output.append(inst)
                else:
                    # tag is probably keywords or input
                    # count the number of commas
                    num_commas = item.count(",")
                    if num_commas >= 5:
                        tag = "keywords"
                    else:
                        tag = "input"
                    inst = {"text": remove_num(item.strip()), "tag": tag}
                    cur_output.append(inst)

        all_results.append(cur_output)
    return all_results


def parens(outputs: list, **kwargs) -> list:
    all_results = []
    for e_idx, entry in enumerate(outputs):
        cur_output = []
        for item in entry.split("\n"):
            if item.strip() != "":
                # get value in parens
                if "(" in item:
                  answer = item.split("(")[-1].split(")")[0]
                else:
                  print(item)
                  continue
                  raise Exception("No parens")
              
                question = item.split("(")[0].strip()
                if not len(question.strip()):
                   continue
                cur_output.append({"question": remove_num(question), "answer": answer})
        all_results.append(cur_output)
    return all_results


def json_parse(outputs: list, **kwargs) -> list:
    all_results = []
    for e_idx, entry in enumerate(outputs):
        cur_output = []
        for item in entry.split("\n"):
            if item.strip() != "":
                try:
                    item_object = ast.literal_eval(item.strip())
                    all_results.append(item_object)
                except Exception as e:
                    print(item_object)
                    print(e)
    return all_results

def strip(outputs: list, **kwargs) -> list:
    all_results = []
    for e_idx, entry in enumerate(outputs):
        if "New Document:" in entry:
            entry = entry.split("New Document:")[-1]
        all_results.append({"text": entry.strip()})
    return all_results


def is_ending_sentence_cot(sentence: str) -> bool:
    list_of_final_sentences = [
        "the answer is",
        "the final answer",
        "so the final answer",
        "answer is",
        "answer:"
    ]
    for item in list_of_final_sentences:
        if item in sentence.lower():
            return True
    return False

def strip_before_rationale(entry: str) -> str:
    if "Rationale:" in entry:
        return entry.split("Rationale:")[-1]
    return entry

def strip_cot(outputs: list, **kwargs) -> list:
    all_results = []
    for e_idx, entry in enumerate(outputs):
        entry = strip_before_rationale(entry)
        last_sent = sent_tokenize(entry)[-1]
        others = sent_tokenize(entry)[:-1]
        if is_ending_sentence_cot(last_sent):
            all_results.append({"text": entry.replace(last_sent, "").strip().replace("Rationale: ", "")})
        else:
            all_results.append({"text": entry.strip().replace("Rationale: ", "")})

    return all_results


def convert_to_ir(output: list, dataset_name: str, write_output_only_to: str, **kwargs) -> list:
    all_results = []
    ids = kwargs["ids"]
    if "tag" in write_output_only_to:
        tag = "question" if "question" in write_output_only_to.lower() else "keyword"
        for idx, item_id in enumerate(ids):
            if idx >= len(output): # debug situations
                break

            # find the correct tag
            text_tags = [item for item in output[idx] if tag in item["tag"].lower()]
            if len(text_tags) == 0:
                print(f"Found {len(text_tags)} tags for {item_id} in {output[idx]}: {text_tags}")
                new = {
                    "_id": item_id,
                    "text": "",
                    "metadata": {"note": "failed to parse"}
                }
                all_results.append(new)
                continue


            # check for edge cases
            if len(text_tags) > 1 and tag == "keyword":
                text_tags = [item for item in text_tags if "question" not in item["tag"].lower()]
            if len(text_tags) > 1 and tag == "keyword":
                text_tags = [item for item in text_tags if item["text"].count(",") >= 3]
            if len(text_tags) > 1 and tag == "question":
                text_tags = [item for item in text_tags if item["text"].count("?")]

            try:
                assert len(text_tags) == 1, f"Found {len(text_tags)} tags for {item_id} in {output[idx]}: {text_tags}"
            except Exception:
                print("Failed on ", item_id)
                new = {
                    "_id": item_id,
                    "text": "",
                    "metadata": {"note": "failed to parse"}
                }
                all_results.append(new)
                continue

            # if text_tags[0]["text"].strip() == "":
            #     breakpoint()

            new = {
                "_id": item_id,
                "text": text_tags[0]["text"],
                "metadata": {}
            }
            all_results.append(new)
    elif "expansion" in write_output_only_to:
        for idx, item in enumerate(ids):
            if idx >= len(output): # debug situations
                break
            new = {
                "_id": item,
                "text": output[idx]["text"]
            }
            all_results.append(new)
    else:
        docs = kwargs["corpus"]
        data = filter_docs_by_run(docs, dataset_name, num_docs=10000)
        tag = "text"
        for idx, item in enumerate(data):
            assert item["doc_id"] in ids[idx], f"Found {ids[idx]} but expected {item['doc_id']}"
            if idx >= len(output):
                break
            if type(output[idx]) == dict:
                item["text"] = output[idx][tag]
            elif type(output[idx]) == list:
                text_tags = [item["text"] for item in output[idx]]
                item["text"] = "\n".join(text_tags)
            else:
                raise NotImplementedError(f"Unknown type {type(output[idx])} for {output[idx]}")

            item["doc_id"] = ids[idx]
            all_results.append(item)

    assert len(all_results) == len(output), f"Found {len(all_results)} results for {dataset_name} but expected {len(output)}"
    return all_results



POSTPROCESS_FNS = {
    "newline_colon": newline_colon,
    "newlines_and_colon": newlines_and_colon,
    "parens": parens,
    "json": json_parse,
    "strip": strip,
    "strip_cot": strip_cot,
    "convert_to_ir": convert_to_ir,
}


   
PREPROCESS_FNS = {
    "jsonl_text": jsonl_text,
    "jsonl_title_and_text_combined": jsonl_title_and_text_combined,
    "jsonl_title_and_text_separate": jsonl_title_and_text_separate,
    "jsonl_title_and_text_newline": jsonl_title_and_text_newline,
    "from_corpus": from_corpus,
    "from_query": from_query,
    "from_query_and_doc": from_query_and_doc,
    "from_list": from_list,
    "filter_docs_by_run": filter_docs_by_run,
    "filter_docs_by_run_300": filter_docs_by_run_300,
    "filter_docs_by_run_1k": filter_docs_by_run_1k,
    "filter_docs_by_run_10k": filter_docs_by_run_10k,
    "filter_queries_by_run": filter_queries_by_run,
    "filter_queries_and_docs_by_run": filter_queries_and_docs_by_run,
    "filter_docs_by_run_300": filter_docs_by_run_300,
    "filter_queries_by_run_300": filter_queries_by_run_300,
    "filter_queries_and_docs_by_run_300_100": filter_queries_and_docs_by_run_300_100,
    "filter_queries_and_docs_by_run_prf": filter_queries_and_docs_by_run_prf,
    "filter_queries_and_docs_by_run_prf_1k": filter_queries_and_docs_by_run_prf_1k,
    "filter_queries_and_docs_by_run_prf_10k": filter_queries_and_docs_by_run_prf_10k,
    "filter_queries_and_docs_by_run_prf_by_id": filter_queries_and_docs_by_run_prf_by_id,
}