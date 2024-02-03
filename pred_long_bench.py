import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

os.environ["WANDB_DISABLED"] = "true"
# from models.modeling_llama import (
#     LlamaForCausalLM as LlamaForCausalLMQuant,
# )
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)

from utils.process_args import process_args
from transformers import LlamaConfig, FalconConfig, MptConfig, MistralConfig, AutoTokenizer
from quant.load_awq_model import load_saved_awq_model

import deepspeed

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt).input_ids[0]
        prompt = tokenizer.decode(prompt[2:], skip_special_tokens=True)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

# def load_model_and_tokenizer(path, model_name, device):
#     if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
#         tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
#     elif "llama2" in model_name:
#         replace_llama_attn_with_flash_attn()
#         tokenizer = LlamaTokenizer.from_pretrained(path)
#         model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
#     elif "longchat" in model_name or "vicuna" in model_name:
#         from fastchat.model import load_model
#         replace_llama_attn_with_flash_attn()
#         model, _ = load_model(
#             path,
#             device='cpu',
#             num_gpus=0,
#             load_8bit=False,
#             cpu_offloading=False,
#             debug=False,
#         )
#         model = model.to(device)
#         model = model.bfloat16()
#         tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
#     model = model.eval()
#     return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    # args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_name = args.model

    # define your model
    model_args, data_args, training_args = process_args()
    # print(model_args, data_args, training_args)
    model_name = model_args.model_name_or_path.split("/")[-1]
    # dtype = torch.bfloat16 if training_args.bf16 else torch.float
    dtype = torch.float16
    q_config = {
        "zero_point": True,  # by default True
        "q_group_size": 128,  # whether to use group quantization
    }
    if 'llama' in model_args.model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True, 
                                            tokenizer_type='llama')
                                            # model_max_length=training_args.model_max_length)
    elif 'falcon' in model_args.model_name_or_path.lower():
        config = FalconConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True)
                                            # model_max_length=training_args.model_max_length)
    elif 'mistral' in model_args.model_name_or_path.lower():
        config = MistralConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True)
                                            # model_max_length=training_args.model_max_length)
    else:
        raise NotImplementedError
    if 'llama' in model_args.model_name_or_path.lower():
        if data_args.use_our_imp:
            print('#' * 50 + 'Use our KV cache quantization' + '#' * 50) 
            from models.modeling_llama_quant import LlamaForCausalLM

            config.k_bits = model_args.k_bits
            config.v_bits = model_args.v_bits
            config.group_size = model_args.group_size
            config.buffer_length = model_args.buffer_length
        else:
            print('#' * 50 + 'Use original Llama implementation' + '#' * 50) 
            from models.modeling_llama import LlamaForCausalLM

        config.attention_dropout = 0.0
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        
        # torch.distributed.init_process_group()
        # model = deepspeed.init_inference(
        #     model=model,      # Transformers models
        #     mp_size=torch.cuda.device_count(),        # Number of GPU
        #     dtype=torch.float16, # dtype of the weights (fp16)
        #     replace_method="auto", # Lets DS autmatically identify the layer to replace
        #     replace_with_kernel_inject=False, # replace the model with the kernel injector
        # )
    elif 'falcon' in model_args.model_name_or_path.lower():
        if data_args.use_our_imp:
            print('#' * 50 + 'Use our KV cache quantization' + '#' * 50) 
            from models.modeling_falcon_quant import FalconForCausalLM

            config.k_bits = model_args.k_bits
            config.v_bits = model_args.v_bits
            config.group_size = model_args.group_size
            config.buffer_length = model_args.buffer_length
        else:
            print('#' * 50 + 'Use original Falcon implementation' + '#' * 50) 
            from models.modeling_falcon import FalconForCausalLM
            
        model = FalconForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    elif 'mistral' in model_args.model_name_or_path.lower():
        if data_args.use_our_imp:
            print('#' * 50 + 'Use our KV cache quantization' + '#' * 50) 
            from models.modeling_mistral_quant import MistralForCausalLM

            config.k_bits = model_args.k_bits
            config.v_bits = model_args.v_bits
            config.group_size = model_args.group_size
            config.buffer_length = model_args.buffer_length
        else:
            print('#' * 50 + 'Use original Mistral implementation' + '#' * 50) 
            from models.modeling_mistral import MistralForCausalLM
            
        model = MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        
    if model_args.load_quant:
        model = load_saved_awq_model(model, model_args.load_quant, config, dtype, model_args.w_bit, q_config)
    #
    # Load model directly
    # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    # model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K")

    model.eval()
    max_length = model2maxlen[model_name]
    if data_args.e:
        # datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
        #     "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        # datasets = ["triviaqa", "qasper", "trec", "samsum"]
        datasets = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "gov_report", "qmsum", "multi_news"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        datasets = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "gov_report", "qmsum", "multi_news"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if data_args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_buffer{model_args.buffer_length}"):
                os.makedirs(f"pred_e/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_buffer{model_args.buffer_length}")
            out_path = f"pred_e/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_buffer{model_args.buffer_length}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_buffer{model_args.buffer_length}"):
                os.makedirs(f"pred/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_buffer{model_args.buffer_length}")
            out_path = f"pred/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_buffer{model_args.buffer_length}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')