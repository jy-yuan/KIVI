# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import torch
import json
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset

config = LlamaConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
config.k_bits = 2 # KiVi currently support 2/4 K/V bits
config.v_bits = 2
config.group_size = 32 
config.residual_length = 32 # corresponding to the number of recent fp16 tokens
config.use_flash = True # use flash-attention with KiVi for long context inference
CACHE_DIR = "/scratch/cached_model"

model = LlamaForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    config=config,
    # cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

enc = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", 
    use_fast=False, 
    trust_remote_code=True,)

model.eval()
file_name = "passkey_examples.jsonl"
method_name = f"K{config.k_bits}V{config.v_bits} KiVi"
print("=========="*2 + f"**{method_name}**" + "=========="*2)
for line in open(file_name, "r"):
    example = json.loads(line)
    prompt_postfix = "What is the pass key? The pass key is "
    prompt = example["input"] + prompt_postfix
    input_ids = enc(prompt, return_tensors="pt").input_ids.cuda()
    print( "-----------------------------------" )
    print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
    print( "Passkey target:", example["target"] )

    tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
    answer = prompt_postfix + enc.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
    answer = answer.replace("\n", "\\n")
    answer= f"{method_name}:\n     [ {answer} ]"
    print( answer )
    print( "-----------------------------------\n" )
