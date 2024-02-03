import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
import argparse
import json, tqdm
import torch
import copy

import math
import time
from lm_eval import evaluator, utils
from lm_eval.tasks import initialize_tasks, include_path
# from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from lm_eval.api.registry import ALL_TASKS

# from models.modeling_llama_quant_backup import LMEvalLlamaForCausalLM
# from models.modeling_llama_quant import LMEvalLlamaForCausalLM
# from models.modeling_llama import LMEvalLlamaForCausalLM
from utils.process_args import process_args
from transformers import LlamaConfig, AutoTokenizer, FalconConfig, MistralConfig
from quant.load_awq_model import load_saved_awq_model
from utils.data import set_seed
from datasets import load_dataset

from accelerate import Accelerator
accelerator = Accelerator()

if __name__ == '__main__':

    set_seed(42)

    model_args, data_args, training_args = process_args()
    # print(model_args, data_args, training_args)
    # dtype = torch.bfloat16 if training_args.bf16 else torch.float
    dtype = torch.float16
    if 'llama' in model_args.model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True, 
                                            tokenizer_type='llama',
                                            model_max_length=training_args.model_max_length)
    elif 'falcon' in model_args.model_name_or_path.lower():
        config = FalconConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True,
                                            model_max_length=training_args.model_max_length)
    elif 'mistral' in model_args.model_name_or_path.lower():
        config = MistralConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=True, 
                                            trust_remote_code=True,
                                            model_max_length=training_args.model_max_length)
    else:
        raise NotImplementedError
    # config.k_bits = model_args.k_bits
    # config.v_bits = model_args.v_bits
    # config.group_size = model_args.group_size
    # model = LlamaForCausalLMQuant.from_pretrained(
    #     pretrained_model_name_or_path=model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    #     torch_dtype=dtype,
    #     low_cpu_mem_usage=False,
    # )
    if torch.cuda.device_count() > 1:
        parallel = True
        low_cpu_mem_usage=True
    else:
        parallel = False
        low_cpu_mem_usage=True
    if 'llama' in model_args.model_name_or_path.lower():
        if data_args.use_our_imp:
            print('#' * 50 + 'Use our KV cache quantization' + '#' * 50) 
            from models.modeling_llama_quant import LMEvalLlamaForCausalLM

            # config.k_bits = model_args.k_bits
            # config.v_bits = model_args.v_bits
            # config.group_size = model_args.group_size
            # model = LlamaForCausalLM.from_pretrained(
            #     pretrained_model_name_or_path=model_args.model_name_or_path,
            #     config=config,
            #     cache_dir=training_args.cache_dir,
            #     torch_dtype=dtype,
            #     low_cpu_mem_usage=True,
            # )
        else:
            print('#' * 50 + 'Use original Llama implementation' + '#' * 50) 
            from models.modeling_llama import LMEvalLlamaForCausalLM

            # config.attention_dropout = 0.0
            # model = LlamaForCausalLM.from_pretrained(
            #     pretrained_model_name_or_path=model_args.model_name_or_path,
            #     config=config,
            #     cache_dir=training_args.cache_dir,
            #     torch_dtype=dtype,
            #     low_cpu_mem_usage=True,
            # )
        model = LMEvalLlamaForCausalLM(
            k_bits=model_args.k_bits,
            v_bits=model_args.v_bits,
            group_size=model_args.group_size,
            buffer_length=model_args.buffer_length,
            pretrained=model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            # parallelize=parallel
        )
    elif 'falcon' in model_args.model_name_or_path.lower():
        if data_args.use_our_imp:
            print('#' * 50 + 'Use our KV cache quantization' + '#' * 50) 
            from models.modeling_falcon_quant import LMEvalFalconForCausalLM
        else:
            print('#' * 50 + 'Use original Falcon implementation' + '#' * 50) 
            from models.modeling_falcon import LMEvalFalconForCausalLM
        model = LMEvalFalconForCausalLM(
            k_bits = model_args.k_bits,
            v_bits = model_args.v_bits,
            group_size = model_args.group_size,
            buffer_length=model_args.buffer_length,
            pretrained=model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            # parallelize=parallel
        )
    elif 'mistral' in model_args.model_name_or_path.lower():
        if data_args.use_our_imp:
            print('#' * 50 + 'Use our KV cache quantization' + '#' * 50) 
            from models.modeling_mistral_quant import LMEvalMistralForCausalLM
        else:
            print('#' * 50 + 'Use original Mistral implementation' + '#' * 50) 
            from models.modeling_mistral import LMEvalMistralForCausalLM
        model = LMEvalMistralForCausalLM(
            k_bits = model_args.k_bits,
            v_bits = model_args.v_bits,
            group_size = model_args.group_size,
            buffer_length=model_args.buffer_length,
            pretrained=model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            # parallelize=parallel
        )
    else:
        raise NotImplementedError
    # model = model.eval().cuda()

    if model_args.load_quant:
        q_config = {
        "zero_point": True,  # by default True
        "q_group_size": 128,  # whether to use group quantization
        }
        model = load_saved_awq_model(model, model_args.load_quant, config, dtype, model_args.w_bit, q_config)

    print("#" * 50 + f"batch size: {data_args.batch_size}" + "#" * 50)
    if data_args.tasks is not None:
        initialize_tasks()
        tasks_list = data_args.tasks.split(",")
        task_names = utils.pattern_match(tasks_list, ALL_TASKS)
        for task in [task for task in tasks_list if task not in task_names]:
            if os.path.isfile(task):
                config = utils.load_yaml_config(task)
                task_names.append(config)
        task_missing = [
            task
            for task in tasks_list
            if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used

        if task_missing:
            missing = ", ".join(task_missing)
            raise ValueError(
                f"Tasks {missing} were not found. Try `lm-eval --tasks list` for list of available tasks."
            )
        results = evaluator.simple_evaluate(
            model=model,
            # model_args='parallelize=True',
            tasks=task_names,
            batch_size=data_args.batch_size,
            log_samples=True
            # no_cache=True,
            # num_fewshot=data_args.num_fewshot,
        )
        print(evaluator.make_table(results))
        samples = results["samples"]
        filepath = f"./output_samples/{training_args.exp_name}.json"
        with open(filepath, "w") as f:
            json.dump(samples, f)
        # if data_args.output_path is not None:
        #     os.makedirs(os.path.dirname(data_args.output_path), exist_ok=True)
        #     # otherwise cannot save
        #     results["config"]["model"] = model_args.model_name_or_path
        #     with open(data_args.output_path, "w") as f:
        #         json.dump(results, f, indent=2)