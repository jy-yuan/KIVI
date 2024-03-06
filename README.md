# KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

Implementation of [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)

## Updates

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
conda create --name <env> --file requirements.txt
```

To install our CUDA implementation:

```bash
cd quant && pip install -e .
```

### Example

Load KIVI-quantized model.

```python
# LLaMA model with KIVI
from models.llama_kivi import LlamaForCausalLM_KIVI

config.k_bits = K_BITS
config.v_bits = V_BITS
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH

model = LlamaForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-2-7b-hf',
    config=config,
    cache_dir=training_args.cache_dir,
    torch_dtype=dtype,
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

Evaluate KIVI on LongBench.

```bash
bash scripts/long_test.sh {GPU_ID} {K_BITS} {V_BITS} {GROUP_LENGTH} {RESIDUAL_LENGTH} {MODEL_NAME}
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
