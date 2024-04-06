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
from lm_eval.api.registry import ALL_TASKS

from utils.process_args import process_args
from transformers import LlamaConfig, AutoTokenizer, FalconConfig, MistralConfig
from utils.data import set_seed
from datasets import load_dataset

from accelerate import Accelerator
accelerator = Accelerator()

if __name__ == '__main__':

    set_seed(42)

    model_args, data_args, training_args = process_args()
    dtype = torch.float16
    if 'llama' in model_args.model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True, 
                                            tokenizer_type='llama',
                                            model_max_length=training_args.model_max_length)
    else:
        raise NotImplementedError

    if torch.cuda.device_count() > 1:
        parallel = True
        low_cpu_mem_usage=True
    else:
        parallel = False
        low_cpu_mem_usage=True
    if 'llama' in model_args.model_name_or_path.lower():
        if model_args.k_bits == 16 and model_args.v_bits == 16:
            from models.modeling_llama import LMEvalLlamaForCausalLM
            model = LMEvalLlamaForCausalLM(
                k_bits=model_args.k_bits,
                v_bits=model_args.v_bits,
                group_size=model_args.group_size,
                residual_length=model_args.residual_length,
                pretrained=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                dtype=dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
        else:
            assert model_args.k_bits in [2, 4] and model_args.v_bits in [2, 4]
            from models.llama_kivi import LMEvalLlamaForCausalLM_KIVI

            model = LMEvalLlamaForCausalLM_KIVI(
                k_bits=model_args.k_bits,
                v_bits=model_args.v_bits,
                group_size=model_args.group_size,
                residual_length=model_args.residual_length,
                pretrained=model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                dtype=dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
    else:
        raise NotImplementedError
    # model = model.eval().cuda()

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
            log_samples=True
            # no_cache=True,
            # num_fewshot=data_args.num_fewshot,
        )
        print(evaluator.make_table(results))
        # samples = results["samples"]
        # filepath = f"./output_samples/{training_args.exp_name}.json"
        # with open(filepath, "w") as f:
        #     json.dump(samples, f)
        # if data_args.output_path is not None:
        #     os.makedirs(os.path.dirname(data_args.output_path), exist_ok=True)
        #     # otherwise cannot save
        #     results["config"]["model"] = model_args.model_name_or_path
        #     with open(data_args.output_path, "w") as f:
        #         json.dump(results, f, indent=2)