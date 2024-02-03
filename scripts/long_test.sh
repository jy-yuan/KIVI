
#meta-llama/Llama-2-7b-chat-hf
#huggyllama/llama-7b
gpuid=$1
k_bits=$2
v_bits=$3
group_size=$4
buffer_length=$5
model=$6
our_imp=$7
e=0

CUDA_VISIBLE_DEVICES=$gpuid python pred_long_bench.py --model_name_or_path $model \
    --tasks none \
    --batch_size 1 \
    --num_fewshot $shot \
    --output_path outputs \
    --model_max_length 4096 \
    --cache_dir ./cached_models \
    --num_train_epochs 20 \
    --n_train_samples 1000 \
    --n_eval_samples 100 \
    --qat \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --buffer_length $buffer_length \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --do_train \
    --do_eval \
    --use_our_imp ${our_imp} \
    --e ${e}