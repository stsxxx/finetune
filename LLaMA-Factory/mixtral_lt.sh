# export PATH=/home/stilex/.local/cuda/bin:$PATH
# export HF_HOME=/data3/llama_model/yuchen
# export HF_HOME=/data6/stilex

# export TMPDIR=/home/stilex/temp
# export PATH=/home/stilex/.local/bin:$PATH

mkdir -p profile_data/mixtral/latency

# change it to your transformers library path i.e. /home/xxx/.local/lib/python3.8/site-packages/transformers
transformers_path="xxxxx"
# change it to the huggingface hub path where the Mixtral model config is stored (config json file path)i.e. "/xxxx/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/521a77772f0d4052fd9846846471d0d2517739d2/config.json"
config_file_path="xxxxx"


cp -f ../copy_for_prof/modeling_mixtral.py "$transformers_path/models/mixtral"
cp -f ../copy_for_prof/trainer.py $transformers_path
cp -f ../copy_for_tune/pytorch_utils.py $transformers_path
cp -f ../copy_for_prof/workflow.py ./src/llmtuner/train/sft
cp -f ../copy_for_tune/utils.py ./src/llmtuner/data


# Specify the new value for "num_experts_per_tok"
new_value=2  # Change this to the desired value

# Use jq to update the value of "num_experts_per_tok" in the JSON file
sed -i.bak "s/\"num_experts_per_tok\":.*/\"num_experts_per_tok\": $new_value,/" "$config_file_path"


echo "Updated 'num_experts_per_tok' to $new_value in $config_file_path"


json_file_path="../copy_for_prof/trainer.py"
# List of batch sizes to iterate over
batch_sizes=(1 10 32)
seq_len=128

for batch_size in "${batch_sizes[@]}"; do
    # Update the BATCH_SIZE in the Python file
    sed -i.bak "s/^BATCH_SIZE = [0-9]\+$/BATCH_SIZE = $batch_size/" "$json_file_path"
    sed -i.bak "s/^SEQ_LEN = [0-9]\+$/SEQ_LEN = $seq_len/" "$json_file_path"

    cp -f ../copy_for_prof/trainer.py "$transformers_path"
    
    # Run the training command with the updated batch size
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
        --per_device_train_batch_size 1 \
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
        --bf16 > ./profile_data/mixtral/latency/sparseb${batch_size}_${seq_len}.txt 

    # /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu -i ./profile_data/mixtral/ncu/sparseb${batch_size}_${seq_len}.ncu-rep > ./profile_data/mixtral/ncu/sparseb${batch_size}_${seq_len}ncu.txt 
    # Optionally wait for the process to complete before starting the next iteration
    wait
done

new_value=8  # Change this to the desired value

# Use jq to update the value of "num_experts_per_tok" in the JSON file
sed -i.bak "s/\"num_experts_per_tok\":.*/\"num_experts_per_tok\": $new_value,/" "$config_file_path"


echo "Updated 'num_experts_per_tok' to $new_value in $config_file_path"


batch_sizes=(1 10)
seq_len=128

for batch_size in "${batch_sizes[@]}"; do
    # Update the BATCH_SIZE in the Python file
    sed -i.bak "s/^BATCH_SIZE = [0-9]\+$/BATCH_SIZE = $batch_size/" "$json_file_path"
    sed -i.bak "s/^SEQ_LEN = [0-9]\+$/SEQ_LEN = $seq_len/" "$json_file_path"

    cp -f ../copy_for_prof/trainer.py "$transformers_path"
    
    # Run the training command with the updated batch size
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
        --per_device_train_batch_size 1 \
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
        --bf16 > ./profile_data/mixtral/latency/denseb${batch_size}_${seq_len}.txt 

    # /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu -i ./profile_data/mixtral/ncu/denseb${batch_size}_${seq_len}.ncu-rep > ./profile_data/mixtral/ncu/denseb${batch_size}_${seq_len}ncu.txt 
    # Optionally wait for the process to complete before starting the next iteration
    wait
done
