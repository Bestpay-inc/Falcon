import json
from transformers import AutoTokenizer
import numpy as np

tokenizer=AutoTokenizer.from_pretrained("Llama-2-7b-chat-hf")


jsonl_file = 'falcon-llama-7b-oase_batch5-temperature-0.0.jsonl'
jsonl_file = 'falcon-llama-7b-ce_layer4-temperature-0.0.jsonl'
jsonl_file_base = 'mt_bench/baseline_collections/base-llama-7b-kongka7-1107-temperature-0.0.jsonl'
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

speeds=[]
acc_token=0
tot_token=0
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    tot_token+=tokens
    try:
        acc_token+=sum(datapoint["choices"][0]["accept tokens"])
    except KeyError:
        acc_token+=sum(datapoint["choices"][0]["accept_tokens"])
    speeds.append(tokens/times)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens


print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())
print(f'Accept rate: {acc_token/tot_token}')
