import json
import transformers
from transformers import AutoTokenizer
from typing import Dict, Optional, List
from transformers.trainer_pt_utils import LabelSmoother
import torch

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    filtered_sources = list()
    res_num = 0
    for i, item in enumerate(sources):
        source = item["conversations"]
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        if len(input_id) <= max_len:
            res_num+=1
            print(res_num)
            filtered_sources.append(item)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))

        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    return dict(dataset=filtered_sources)

def merge_all_data():
    with open("/root/LLM-based-graph-tool/data/datasets/InternVL2_flowchart_Dataset/vbase1000/dataset/base_graph_data.json", 'r', encoding="utf-8") as f:
        base_graph_data = json.load(f)
    with open("/root/LLM-based-graph-tool/data/datasets/internvl_frdetr/internvl_frdetr_tain_dataset_v2.json", 'r', encoding="utf-8") as f:
        v2_frdetr_train = json.load(f)
    mid = len(base_graph_data)
    for item in v2_frdetr_train:
        item["id"]+=mid
        item["image"] = item["image"].replace("/root/LLM-based-graph-tool/data/datasets/frdetr_dataset/train", "/root/LLM-based-graph-tool/data/datasets/InternVL2_flowchart_Dataset/vbase1000/Images")
        base_graph_data.append(item)
    # base_graph_data.extend(v2_frdetr_train)
    # print(f"base_graph_data len:{len(base_graph_data)}")
    # res_datasets = list()
    # res_datasets.extend(base_graph_data[:1000])
    # res_datasets.extend(v2_frdetr_train[:1000])

    # max_len = 0
    # for item in res_datasets:
    #     if len(str(item)) > max_len: max_len=len(str(item))
    # print(max_len)
    with open("/root/LLM-based-graph-tool/data/datasets/InternVL2_flowchart_Dataset/vbase1000/dataset/internvl2_train_dataset.json", 'w', encoding="utf-8") as f:
        json.dump(base_graph_data, f, ensure_ascii=False)
    
def filter_all_data():
    merged_data_path = "/root/LLM-based-graph-tool/data/datasets/train_dataset_v3.json"
    model_name_or_path = "Qwen/Qwen-VL-Chat"
    model_max_length = 4096
    with open(merged_data_path, "r", encoding="utf-8") as f:
        merged_dataset = json.load(f)
    # sources = [example["conversations"] for example in merged_dataset]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    filtered_datasets = preprocess(merged_dataset, tokenizer, model_max_length)

    with open("/root/LLM-based-graph-tool/data/datasets/filtered_train_dataset.json", 'w', encoding="utf-8") as f:
        json.dump(filtered_datasets["dataset"], f, ensure_ascii=False)
    

def verify_base_graph_datasets():
    base_graph_dataset_path = "/root/LLM-based-graph-tool/data/datasets/internvl_baseGraphV1/dataset/base_graph_data.json"
    

def main():
    merge_all_data()

if __name__== "__main__":
    main()