# export PATH=/home/stilex/.local/cuda/bin:$PATH
# export HF_HOME=/data3/llama_model/yuchen
# export HF_HOME=/data6/stilex
# export TMPDIR=/home/stilex/temp
# export PATH=/home/stilex/.local/bin:$PATH

mkdir -p profile_data/mixtral/throughput

# change it to your transformers library path i.e. /home/xxx/.local/lib/python3.8/site-packages/transformers
transformers_path="xxxxx"
# change it to the huggingface hub path where the Mixtral model config is stored (config json file path) i.e. "/xxxx/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/521a77772f0d4052fd9846846471d0d2517739d2/config.json"
config_file_path="xxxxx"

cp -f ../copy_for_tune/modeling_mixtral.py "$transformers_path/models/mixtral"
cp -f ../copy_for_tune/trainer.py $transformers_path
cp -f ../copy_for_tune/pytorch_utils.py $transformers_path
cp -f ../copy_for_tune/workflow.py ./src/llmtuner/train/sft
cp -f ../copy_for_tune/utils.py ./src/llmtuner/data


# Specify the new value for "num_experts_per_tok"
new_value=2  # Change this to the desired value

# Use jq to update the value of "num_experts_per_tok" in the JSON file
sed -i.bak "s/\"num_experts_per_tok\":.*/\"num_experts_per_tok\": $new_value,/" "$config_file_path"


echo "Updated 'num_experts_per_tok' to $new_value in $config_file_path"

batch_sizes=(1 3)

for batch_size in "${batch_sizes[@]}"; do   
    cp -f ../copy_for_tune/modeling_mixtral.py "$transformers_path/models/mixtral"
    cp -f ../copy_for_tune/trainer.py $transformers_path
    cp -f ../copy_for_tune/workflow.py ./src/llmtuner/train/sft
    cp -f ../copy_for_tune/utils.py ./src/llmtuner/data  
    CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
        --stage sft \
        --do_train \
        --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
        --dataset math \
        --template mistral \
        --finetuning_type lora \
        --lora_target w1,w2,w3,gate \
        --flash_attn \
        --output_dir mixtral \
        --overwrite_output_dir \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_steps 600 \
        --learning_rate 5e-5 \
        --num_train_epochs 1 \
        --quantization_bit 4 \
        --cutoff_len 1024 \
        --lora_rank 16 \
        --bf16 > ./profile_data/mixtral/throughput/math_throughput_b${batch_size}_sparse.txt 
    wait
done

batch_sizes=(1 2 8)

for batch_size in "${batch_sizes[@]}"; do   
    cp -f ../copy_for_tune/modeling_mixtral.py "$transformers_path/models/mixtral"
    cp -f ../copy_for_tune/trainer.py $transformers_path
    cp -f ../copy_for_tune/workflow.py ./src/llmtuner/train/sft
    cp -f ../copy_for_tune/utils.py ./src/llmtuner/data  
    CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
        --stage sft \
        --do_train \
        --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
        --dataset commonsense \
        --template mistral \
        --finetuning_type lora \
        --lora_target w1,w2,w3,gate \
        --flash_attn \
        --output_dir mixtral \
        --overwrite_output_dir \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_steps 600 \
        --learning_rate 5e-5 \
        --num_train_epochs 1 \
        --quantization_bit 4 \
        --cutoff_len 1024 \
        --lora_rank 16 \
        --bf16 > ./profile_data/mixtral/throughput/commonsense_throughput_b${batch_size}_sparse.txt 
    wait
done


new_value=8  # Change this to the desired value

# Use jq to update the value of "num_experts_per_tok" in the JSON file
sed -i.bak "s/\"num_experts_per_tok\":.*/\"num_experts_per_tok\": $new_value,/" "$config_file_path"

echo "Updated 'num_experts_per_tok' to $new_value in $config_file_path"

batch_sizes=(1)

for batch_size in "${batch_sizes[@]}"; do   
    cp -f ../copy_for_tune/modeling_mixtral.py "$transformers_path/models/mixtral"
    cp -f ../copy_for_tune/trainer.py $transformers_path
    cp -f ../copy_for_tune/workflow.py ./src/llmtuner/train/sft
    cp -f ../copy_for_tune/utils.py ./src/llmtuner/data  
    CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
        --stage sft \
        --do_train \
        --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
        --dataset math \
        --template mistral \
        --finetuning_type lora \
        --lora_target w1,w2,w3,gate \
        --flash_attn \
        --output_dir mixtral \
        --overwrite_output_dir \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_steps 600 \
        --learning_rate 5e-5 \
        --num_train_epochs 1 \
        --quantization_bit 4 \
        --cutoff_len 1024 \
        --lora_rank 16 \
        --bf16 > ./profile_data/mixtral/throughput/math_throughput_b${batch_size}_dense.txt 
    wait
done

batch_sizes=(1 2)

for batch_size in "${batch_sizes[@]}"; do   
    cp -f ../copy_for_tune/modeling_mixtral.py "$transformers_path/models/mixtral"
    cp -f ../copy_for_tune/trainer.py $transformers_path
    cp -f ../copy_for_tune/workflow.py ./src/llmtuner/train/sft
    cp -f ../copy_for_tune/utils.py ./src/llmtuner/data  
    CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
        --stage sft \
        --do_train \
        --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
        --dataset commonsense \
        --template mistral \
        --finetuning_type lora \
        --lora_target w1,w2,w3,gate \
        --flash_attn \
        --output_dir mixtral \
        --overwrite_output_dir \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_steps 600 \
        --learning_rate 5e-5 \
        --num_train_epochs 1 \
        --quantization_bit 4 \
        --cutoff_len 1024 \
        --lora_rank 16 \
        --bf16 > ./profile_data/mixtral/throughput/commonsense_throughput_b${batch_size}_dense.txt 
    wait
done
