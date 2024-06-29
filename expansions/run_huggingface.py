from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, T5ForConditionalGeneration, T5Tokenizer, set_seed, LlamaForCausalLM, LlamaTokenizer

import torch
import pandas as pd
import os
import json
import argparse
import tqdm
from torch.utils.data import Dataset              

set_seed(2)

# so that we save large models to a place with enough space 
os.environ["HF_HOME"] = "/home/person/nfs/huggingface_cache"

model_loaded = None
tokenizer = None

class PromptDataset(Dataset):  
    def __init__(self, list_of_examples):
        super().__init__()
        self.examples = list_of_examples

    def __len__(self):                                                              
        return len(self.examples)                                                                 

    def __getitem__(self, i):                                                       
        return self.examples[i]                                                


def predict(input_prompts, model_type="t5", temperature: float = 0.3, specific_name: str = "google/flan-t5-xxl", batch_size: int = 1, **kwargs):
    global model_loaded
    global tokenizer

    dataset = PromptDataset(input_prompts)

    if model_type == "t5":
        # load examples to classify
        print("Loading T5 based model..", specific_name)
        if model_loaded is None:
            tokenizer = T5Tokenizer.from_pretrained(specific_name)
            model = T5ForConditionalGeneration.from_pretrained(specific_name, torch_dtype=torch.bfloat16).cuda()
            text2text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
            model_loaded = model
        else:
            model = model_loaded
            text2text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
    
    elif model_type in ["llama"]:
        if model_loaded is None:
            tokenizer = LlamaTokenizer.from_pretrained(specific_name)
            model = LlamaForCausalLM.from_pretrained(specific_name, torch_dtype=torch.float16).cuda()
            text2text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
            text2text_generator.tokenizer.pad_token_id = model.config.eos_token_id
            model_loaded = model
        else:
            model = model_loaded.cuda()
            text2text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
            text2text_generator.tokenizer.pad_token_id = model.config.eos_token_id
    else:
        text2text_generator = pipeline("text-generation", model=specific_name, device=0)

    # print(f"Num Parameters: {model.num_parameters()}")
    generated_strings = []
    for out in tqdm.tqdm(text2text_generator(input_prompts, batch_size=batch_size, max_length=2048), total=len(dataset)):
        generated_strings.append(item["generated_text"] for item in out)
    assert len(generated_strings) == len(dataset)
    generated_strings = [next(item).replace(input_prompts[i], "") for i, item in enumerate(generated_strings)]
    return list(generated_strings), 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    print(predict(["What is capitol of France?"]))
