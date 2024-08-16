# KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

Implementation of [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)

## Updates
- [2024.06.07]:🎉 KIVI largely inspires the [HuggingFace Transformers KV Cache quantization](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [2024.06.06]:(Beta) We extensively optimize the codebase in [branch develop](https://github.com/jy-yuan/KIVI/tree/develop) to reduce the latency of KIVI. Note that **you need to reinstall our CUDA implementation** under the ```quant``` folder. We will release a blog soon about the detailed optimization.
- [2024.05.01]:🎉 KIVI has been accepted by ICML 2024! See you in Vienna!
- [2024.04.12]: We add the support for Mistral model family. The performance of LongChat-7b-v1.5-32K and Mistral-7B-Instruct-v0.2 on 15 tasks from LongBench can be found in [long_bench.md](./docs/long_bench.md).

- [2024.04.05]: We release the code for reproducing our CoQA/TruthfulQA/GSM8K results using LM-Eval. Please check the [README of branch lmeval](https://github.com/jy-yuan/KIVI/tree/lmeval).

- [2024.04.04]: 🔥🔥We add a new 5-digit [passkey example](./long_context_example.py) with 12k context length to show the performance of 2bit KIVI under the long context senario.

- [2024.04.04]: (Beta) We add the flash-attention support for KIVI during the prefill phase. 

- [2024.04.03]: We add a new [5-shot GSM8K example.py](./example.py) to show the performance of 2/4 bit KIVI with 32 full precision tokens.

- [2024.02.05]: KIVI ver. 2 is released on [arXiv](https://arxiv.org/abs/2402.02750).

- [2024.02.03]: KIVI code is released.

- [2023.12.29]: KIVI ver. 1 is released on [researchgate](https://www.researchgate.net/publication/376831635_KIVI_Plug-and-play_2bit_KV_Cache_Quantization_with_Streaming_Asymmetric_Quantization).

## Overview

KIVI is a new plug-and-play 2bit KV cache quantization algorithm without any fine-tuning. This algorithm optimizes memory usage by quantizing the key cache per-channel and the value cache per-token to 2bit. KIVI's hardware-friendly design allows LLMs like Llama-2, Falcon, and Mistral to maintain comparable quality levels while reducing peak memory usage by 2.6 times. This enables up to 4 times larger batch sizes and significantly increases throughput by 2.35 to 3.47 times in real LLM inference workloads, effectively addressing the bottleneck issues in speed and memory usage.

Illustration of KIVI quantization scheme: key cache per-channel and value cache per-token.
<p align="center">
<img width="300" src="./img/quant_scheme.png">
</p>

Illustration of KIVI algorithm during inference prefill and decoding phase:
<p align="center">
<img width="700" src="./img/algo.png">
</p>

## How to use KIVI

### Setup

To install the required packages:

```bash
conda create -n kivi python=3.10
conda activate kivi
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

Then install our CUDA implementation:

```bash
cd quant && pip install -e .
```

### Example

Load model with KIVI: (e.g., Llama-2-7b)

```python
# LLaMA model with KIVI
import torch
import os
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

config.k_bits = K_BITS # current support 2/4 bit for KV Cache
config.v_bits = V_BITS # current support 2/4 bit for KV Cache
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH # the number of recent fp16 tokens
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

model = LlamaForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-2-7b-hf',
    config=config,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf', 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

# Inference
# e.g., model.generate(...)
```

#### GSM8K example
We use GSM8K as an example to show how to use KIVI. You can check [example.py](./example.py):

```bash
python example.py
```

#### Passkey retrieval example

Passkey retrieval with KIVI. You can check [long_context_example.py](./long_context_example.py):

```bash
python long_context_example.py
```

#### Evaluate KIVI on LongBench

We currently support Llama and Mistral family of models. We recently test KIVI on Mistral-7B-Instruct-v0.2 and Longchat-7b-v1.5-32k. Please check [long_bench.md](./docs/long_bench.md) for more details.
```bash
bash scripts/long_test.sh {GPU_ID} {K_BITS} {V_BITS} {GROUP_LENGTH} {RESIDUAL_LENGTH} {MODEL_NAME}
python eval_long_bench.py --model {MODEL} # MODEL is the dir name under pred/ Currently it support Llama family model and Mistral model.
```

## Citation

If you find our method useful, please kindly cite our paper.

```bibtex
@article{liu2024kivi,
  title={KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache},
  author={Liu, Zirui and Yuan, Jiayi and Jin, Hongye and Zhong, Shaochen and Xu, Zhaozhuo and Braverman, Vladimir and Chen, Beidi and Hu, Xia},
  journal={arXiv preprint arXiv:2402.02750},
  year={2024}
}
```

## Contributing
We welcome contributions from the research community to improve KIVI. If you have any idea or would like to report a bug, please open an issue or submit a pull request.

## License
The code is released under the MIT License.
