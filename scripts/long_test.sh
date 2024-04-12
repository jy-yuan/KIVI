# model e.g.: meta-llama/Llama-2-7b-hf

gpuid=$1
k_bits=$2
v_bits=$3
group_size=$4
residual_length=$5
model=$6
e=0

CUDA_VISIBLE_DEVICES=$gpuid python pred_long_bench.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --residual_length $residual_length \
    --e ${e}