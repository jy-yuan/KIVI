# model=meta-llama/Llama-2-13b-chat-hf
# huggyllama/llama-7b
# meta-llama/Llama-2-7b-hf
gpuid=$1
k_bits=$2
v_bits=$3
group_size=$4
buffer_length=$5
tasks=$6
our_imp=$7
model=$8

model_name="${model#*/}"
echo "$model_name"
CUDA_VISIBLE_DEVICES=$gpuid python run_lm_eval_harness.py --model_name_or_path $model \
    --tasks $tasks \
    --batch_size 128 \
    --num_fewshot 0 \
    --output_path outputs \
    --model_max_length 1024 \
    --cache_dir ./cached_models \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --buffer_length $buffer_length \
    --use_our_imp ${our_imp} \
    --exp_name ${tasks}_${model_name}_k${k_bits}v${v_bits}our_imp${our_imp} # 2>&1 | tee profile_k${k_bits}v${v_bits}.txt &
