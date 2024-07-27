
export PYTHONPATH=$PYTHONPATH:../BlackMamba

mkdir -p profile_data/blackmamba/throughput

# change it to your transformers library path i.e. /home/xxx/.local/lib/python3.8/site-packages/transformers
transformers_path="xxxxx"
# change it to the huggingface hub path where the BlackMamba model config is stored (directory path)i.e. "/xxxx/hub/models--Zyphra--BlackMamba-2.8B/snapshots/521a77772f0d4052fd9846846471d0d2517739d2"
model_path="xxxxx"

cp -f ../copy_for_tune/pytorch_utils.py $transformers_path
cp -f ../copy_for_mamba_tune/trainer.py $transformers_path
cp -f ../copy_for_mamba_tune/workflow.py ./src/llmtuner/train/sft
cp -f ../copy_for_mamba_tune/utils.py ./src/llmtuner/data

config_file_path="../copy_for_mamba/config.json"


# Specify the new value for "topkk"
new_value=2  # Change this to the desired value

sed -i.bak "s/\"topk\":.*/\"topk\": $new_value}/" "$config_file_path"


echo "Updated 'topk' to $new_value in $config_file_path"
cp -f "$config_file_path" "$model_path/"

json_file_path="../copy_for_mamba_tune/trainer.py"
# List of batch sizes to iterate over
batch_sizes=(1 2 8)

for batch_size in "${batch_sizes[@]}"; do


    cp -f ../copy_for_mamba_tune/trainer.py "$transformers_path"
    
    # Run the training command with the updated batch size
    # -c 3000
    CUDA_VISIBLE_DEVICES=0  python src/train_bash.py \
        --stage sft \
        --do_train \
        --model_name_or_path Zyphra/BlackMamba-2.8B \
        --dataset math \
        --template mistral \
        --finetuning_type lora \
        --lora_target w1,w2,w3,gate \
        --output_dir Blackmamba \
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
        --bf16 > ./profile_data/blackmamba/throughput/math_sparse_b${batch_size}.txt 

    # /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu -i /home/stilex/profile_data/mamba/ncu_moe/sparseb${batch_size}_${seq_len}.ncu-rep > /home/stilex/profile_data/mamba/ncu_moe/sparseb${batch_size}_${seq_len}ncu.txt 
    # Optionally wait for the process to complete before starting the next iteration
    wait
done

batch_sizes=(1 6 20)

for batch_size in "${batch_sizes[@]}"; do


    cp -f ../copy_for_mamba_tune/trainer.py "$transformers_path"
    
    # Run the training command with the updated batch size
    # -c 3000
    CUDA_VISIBLE_DEVICES=0  python src/train_bash.py \
        --stage sft \
        --do_train \
        --model_name_or_path Zyphra/BlackMamba-2.8B \
        --dataset commonsense \
        --template mistral \
        --finetuning_type lora \
        --lora_target w1,w2,w3,gate \
        --output_dir Blackmamba \
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
        --bf16 > ./profile_data/blackmamba/throughput/commonsense_sparse_b${batch_size}.txt 


    wait
done


new_value=8  # Change this to the desired value

sed -i.bak "s/\"topk\":.*/\"topk\": $new_value}/" "$config_file_path"


echo "Updated 'topk' to $new_value in $config_file_path"

cp -f "$config_file_path" "$model_path/"


batch_sizes=(1 2)

for batch_size in "${batch_sizes[@]}"; do


    cp -f ../copy_for_mamba_tune/trainer.py "$transformers_path"
    
    # Run the training command with the updated batch size
    # -c 3000
    CUDA_VISIBLE_DEVICES=0  python src/train_bash.py \
        --stage sft \
        --do_train \
        --model_name_or_path Zyphra/BlackMamba-2.8B \
        --dataset math \
        --template mistral \
        --finetuning_type lora \
        --lora_target w1,w2,w3,gate \
        --output_dir Blackmamba \
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
        --bf16 > ./profile_data/blackmamba/throughput/math_dense_b${batch_size}.txt 


    wait
done



batch_sizes=(1 6)

for batch_size in "${batch_sizes[@]}"; do


    cp -f ../copy_for_mamba_tune/trainer.py "$transformers_path"
    
    # Run the training command with the updated batch size
    # -c 3000
    CUDA_VISIBLE_DEVICES=0  python src/train_bash.py \
        --stage sft \
        --do_train \
        --model_name_or_path Zyphra/BlackMamba-2.8B \
        --dataset commonsense \
        --template mistral \
        --finetuning_type lora \
        --lora_target w1,w2,w3,gate \
        --output_dir Blackmamba \
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
        --bf16 > ./profile_data/blackmamba/throughput/commonsense_dense_b${batch_size}.txt 


    wait
done
