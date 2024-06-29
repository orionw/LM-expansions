from argparse import ArgumentParser
from langchain.chat_models import ChatAnthropic
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import logging
import sys
import asyncio
import time
import tqdm
import random
import together
import os


# if needed
together.api_key = os.environ.get("TOGETHER_API_KEY")
    


logger = logging.getLogger(__name__)




async def predict(input_prompts, model_type='chatgpt', temperature: float = 0.3, specific_name: str = "text-davinci-003", system_prompt: str = None, **kwargs):
    # batch over prompts in batches of 3000
    all_generations = []
    total_tokens_all = 0
    batch_num = 100 if model_type != "chatgpt" else 300
    if model_type == "together":
        batch_num = 10000000
    for i in range(0, len(input_prompts), batch_num):
        print("Batching for prompt", i, "to", i+batch_num, "of", len(input_prompts), file=sys.stderr, flush=True)
        batch = input_prompts[i:i+batch_num]
        generations, total_tokens = await predict_batch(batch, model_type=model_type, temperature=temperature, specific_name=specific_name, system_prompt=system_prompt, **kwargs)
        all_generations.extend(generations)
        total_tokens_all += total_tokens
        if model_type == "gpt4" and i + batch_num < len(input_prompts):
            print("Done with batch, sleeping...")
            time.sleep(60)
        if model_type == "chatgpt" and i + batch_num < len(input_prompts):
            print("Done with batch, sleeping...")
            time.sleep(45)
        if model_type == "claude":
            print("Done with batch, sleeping...")
            time.sleep(random.uniform(1.5, 6.5))
        if model_type == "together":
            pass
            # print("Done with batch, sleeping...")
            # time.sleep(1)

    return all_generations, total_tokens



async def predict_batch(input_prompts, model_type='chatgpt', temperature: float = 0.3, specific_name: str = "text-davinci-003", system_prompt: str = None, **kwargs):
    if "batch_size" in kwargs:
        del kwargs["batch_size"]

    if model_type in ["gpt4", 'chatgpt']:
        llm = ChatOpenAI(model_name=specific_name, temperature=temperature, request_timeout=600, max_retries=10)
        if system_prompt is None:
            batch_messages = [[HumanMessage(content=prompt)] for prompt in input_prompts]
        else:
            batch_messages = [[HumanMessage(content=prompt), SystemMessage(content=system_prompt)] for prompt in input_prompts]
        llm_result = await llm.agenerate(batch_messages, **kwargs)

    elif model_type in ['gpt3']:
        llm = OpenAI(model_name=specific_name, temperature=temperature, request_timeout=600, max_retries=10)
        llm_result = await llm.agenerate(input_prompts, **kwargs)
    
    elif model_type in ['claude']:
        llm = ChatAnthropic(model=specific_name, temperature=temperature, request_timeout=600, max_retries=10, max_tokens=2048)
        batch_messages = [[HumanMessage(content=prompt)] for prompt in input_prompts]
        llm_results = []
        tokens_used = 0
        for batch_message in tqdm.tqdm(batch_messages):
            llm_result = llm.generate([batch_message], **kwargs)
            # print(llm_result)
            outputs = [[gen.text.strip() for gen in result] for result in llm_result.generations]
            logging.info(llm_result.llm_output)
            generations = [" ".join(entry) for entry in outputs]
            llm_results.extend(generations)
            time.sleep(random.uniform(1.5, 6.5))

    elif model_type in "together":
        llm_results = []
        tokens_used = 0
        for batch_message in tqdm.tqdm(input_prompts, bar_format='{l_bar}{bar}{r_bar}\n'):
            retries = 3
            while retries > 0:
                try:
                    print(len(batch_message.split(" ")))
                    response = together.Complete.create(model=specific_name, prompt=batch_message, max_tokens=256, temperature=0.3)
                    llm_results.append(response['output']['choices'][0]['text'].strip())
                    if random.random() < 0.1:
                        print(response)
                    break
                except Exception as e:
                    print(f"Error on retry num {3 - retries}: {e}")
                    retries -= 1
                    time.sleep(120)
                    if retries == 0:
                        llm_results.append("")
                        # raise Exception(f"Error'd on retries: {e}")
    else:
        raise NotImplementedError()

    if model_type not in ['claude', "together"]:
        outputs = [[gen.text.strip() for gen in result] for result in llm_result.generations]
        logging.info(llm_result.llm_output)
        generations = [" ".join(entry) for entry in outputs]
        tokens_used = llm_result.llm_output["token_usage"]["total_tokens"]
    else:
        generations = llm_results

    return generations, tokens_used # TODO: not sure why there are multiples here...




if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('')
    # # TODO add prompts
    # args = parser.parse_args()

    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )
    predict(["What is capitol of France?"])

