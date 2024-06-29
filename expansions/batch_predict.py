import os
import pandas as pd
import random
import time
import asyncio
import json
import ast
import copy
import tiktoken

from .run_huggingface import predict as predict_huggingface
from .run_gpt import predict as predict_gpt
from .process_prompts import PREPROCESS_FNS, POSTPROCESS_FNS

enc = tiktoken.get_encoding("cl100k_base")
random.seed(42)

def choose_predict_func(model_name: str):
  if model_name in ["gpt4", "gpt3", "chatgpt", 'claude', "together"]:
    return predict_gpt
  else:
    return predict_huggingface


async def run_examples(prompt: str, model_name: str = "gpt3", specific_name: str = "text-davinci-003", num_examples: int = 2, temperature: float = 0.3, dataset_name: str = "ELI5", desc: str = None, system_prompt: str = None, output_format: str = "semicolon", batch_size: int = 1, input_format: str = "article", write_output_only_to: str = None):
    dataset_name = dataset_name.replace("--", "/") # convert back
    params = {
        "prompt": prompt,
        "desc": desc,
        "system_prompt": system_prompt,
        "model_name": model_name,
        "specific_name": specific_name,
        "num_examples": num_examples,
        "temperature": temperature,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "input_format": input_format,
        "output_format": output_format,
        "write_output_only_to": write_output_only_to # not used currently, but could be used with more options
    }
   
    # define your function before, makes it less messy
    # allow multiple preprocess steps
    corpus = None
    preprocess_funcs = input_format.split("|")
    cur_data = dataset_name
    for func in preprocess_funcs:
        if func not in PREPROCESS_FNS:
            raise Exception(f"Invalid preprocess function {func}")
        preprocess_func = PREPROCESS_FNS[func]
        if func == "from_corpus":
            # save it, since it takes forever to load
            cur_data = preprocess_func(cur_data, dataset_name=dataset_name)
            corpus = copy.deepcopy(cur_data)
        elif func == "from_query_and_doc":
            cur_data = preprocess_func(cur_data, dataset_name=dataset_name, corpus=corpus)
            corpus = copy.deepcopy(cur_data[1])
        else:
            cur_data = preprocess_func(cur_data, dataset_name=dataset_name)

    cur_data, ids = cur_data
    input_data = cur_data

    if corpus is not None and num_examples < len(corpus):
        print("Using debug, shorter num")
        input_data = input_data[30:30+num_examples]

    # allow for arbitrary templates
    number_of_inputs = len(input_data[0]) if len(input_data) and type(input_data[0]) in [tuple, list] else 1
    print(f"Number of inputs: {number_of_inputs}")
    full_prompts = []
    for input_doc in input_data:
        cur_doc = prompt
        for input_idx in range(number_of_inputs):
            cur_input = input_doc[input_idx] if type(input_doc) in [tuple, list] else input_doc
            cur_doc = cur_doc.replace(f"$$$$$${input_idx}", cur_input)  
        full_prompts.append(cur_doc)
        # assert "$$$$$$" not in cur_doc, cur_doc
    
    print(f"Predicting for {len(full_prompts)} examples..")
    print(f"\nSample prompt:\n```\n{full_prompts[0]}```")
    print(f"\nSample prompt2:\n```\n{full_prompts[1]}```")
    # print(f"\nSample prompt3:\n```\n{full_prompts[2]}```")
    # print(f"\nSample prompt4:\n```\n{full_prompts[3]}```")
    # print(f"\nSample prompt5:\n```\n{full_prompts[4]}```")


    ready_prompts = []
    max_tokens = 16000
    for prompt in full_prompts:
      encoding = enc.encode(prompt)
      if len(encoding) > max_tokens:
        ready_prompts.append(enc.decode(encoding[:max_tokens]))
        print("Prompt was too long, may want to check...")
      else:
        ready_prompts.append(prompt)

    full_prompts = ready_prompts

    start_time = time.time()
    tokens_used = 0
    predict = choose_predict_func(model_name)
    outputs, tokens_used = await predict(full_prompts, model_type=model_name, temperature=temperature, specific_name=specific_name, batch_size=batch_size)
    
    postprocess_funcs = output_format.split("|")
    cur_data = outputs
    for func in postprocess_funcs:
        if func not in POSTPROCESS_FNS:
            raise Exception(f"Invalid preprocess function {func}")
        postprocess_func = POSTPROCESS_FNS[func]
        try:
          if func == "convert_to_ir": # we want to save it all
            if "tag" in write_output_only_to:
              end_data_query = postprocess_func(cur_data, ids=ids, full_prompts=full_prompts, dataset_name=dataset_name, write_output_only_to="expansion_question_tag", input_format=input_format, input_data=input_data)
              end_data_keyword = postprocess_func(cur_data, ids=ids, full_prompts=full_prompts, dataset_name=dataset_name, write_output_only_to="expansion_keyword_tag", input_format=input_format, input_data=input_data)
              end_data = [end_data_query, end_data_keyword]
              write_output_only_to = [f"expansion_question--{write_output_only_to}", f"expansion_keyword--{write_output_only_to}"]
            else:
              end_data = [postprocess_func(cur_data, ids=ids, full_prompts=full_prompts, dataset_name=dataset_name, write_output_only_to=write_output_only_to, input_format=input_format, corpus=corpus)]
              write_output_only_to = [write_output_only_to] 
          else:
            cur_data = postprocess_func(cur_data, full_prompts=full_prompts, dataset_name=dataset_name, write_output_only_to=None)
        except Exception as e:
            print(e)
            print(f"Failed to run {func} on {cur_data}")
            cur_data = outputs
            break
        

    all_results = cur_data

    model_time = round(time.time() - start_time, 2)
    combined = {
      "params": params,
      "output": all_results,
      "model_generate_time": model_time,
      "total_tokens_used": tokens_used,
    }
    
    # write out the data to a file based on the current time in `results`
    if desc is None:
       desc = "generic"
    pretty_time_and_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_file = os.path.join(args.output_path, f"{model_name.replace('/', '--')}-{specific_name.replace('/', '--')}-{desc.replace('/', '--')}-{pretty_time_and_date}.json")
    with open(output_file, "w") as f:
        combined["original"] = outputs
        json.dump(combined, f, indent=2)
    print(f"Finished writing to {output_file}")

    
    if write_output_only_to is not None:
      safe_dataset_name = dataset_name.replace("/", "--")
      if not os.path.isdir(os.path.join("llm-based-expansions-generations", safe_dataset_name)):
        os.makedirs(os.path.join("llm-based-expansions-generations", safe_dataset_name))
        
      for end_data_ind, write_output_only_to_ind in zip(end_data, write_output_only_to):
        with open(os.path.join(f"llm-based-expansions-generations", safe_dataset_name, write_output_only_to_ind + f"{len(end_data_ind)}.jsonl"), "w") as f:
            for line in end_data_ind:
                f.write(json.dumps(line) + "\n")
        print(f"Finished writing to {os.path.join(f'llm-based-expansions-generations', safe_dataset_name, write_output_only_to_ind + f'{len(end_data_ind)}.jsonl')}")

      
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--file', help='The JSON file to load what to run', type=str, required=True)
  parser.add_argument('-o', '--output_path', help='Where to write the results to', type=str, default="results")
  args = parser.parse_args()

  if not os.path.isdir(args.output_path):
    os.makedirs(args.output_path)

  df = pd.read_json(args.file, lines=True)
  for idx, row in df.iterrows():
    # no need to use them, they're saved
    print(row)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_examples(row["prompt"], row["model_name"], row["specific_name"], row["num_examples"], row["temperature"], row["dataset_name"], desc=row["desc"] if "desc" in row else None, system_prompt=row["system_prompt"] if "system_prompt" in row else None, output_format=row["output_format"] if "output_format" in row else "semicolon", batch_size=row["batch_size"] if "batch_size" in row else 1, input_format=row["input_format"] if "input_format" in row else "article", write_output_only_to=row["write_output_only_to"] if "write_output_only_to" in row else None))
    