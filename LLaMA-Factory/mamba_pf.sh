# export PATH=/home/stilex/.local/cuda/bin:$PATH
# export HF_HOME=/data6/stilex
# export TMPDIR=/home/stilex/temp
# export PATH=/home/stilex/.local/bin:$PATH
export PATH=../BlackMamba:$PATH

# pip install transformers==4.36.2
# cp -f /home/stilex/copy_for_prof/modeling_mixtral.py /home/stilex/.local/lib/python3.8/site-packages/transformers/models/mixtral
cp -f ../copy_for_mamba/trainer.py /home/stilex/.local/lib/python3.8/site-packages/transformers
cp -f ../copy_for_mamba/workflow.py ./src/llmtuner/train/sft
cp -f ../copy_for_mamba/util.py ./src/llmtuner/data


config_file_path="../copy_for_mamba/config.json"
model_path="/data6/stilex/hub/models--Zyphra--BlackMamba-2.8B/snapshots/521a77772f0d4052fd9846846471d0d2517739d2"
# Specify the new value for "num_experts_per_tok"
new_value=2  # Change this to the desired value

sed -i.bak "s/\"topk\":.*/\"topk\": $new_value,/" "$config_file_path"


echo "Updated 'topk' to $new_value in $config_file_path"

cp -f $config_file_path $model_path
# json_file_path="/data3/llama_model/yuchen/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841/config.json"
# Specify the new value for "num_experts_per_tok"
# new_value=8  # Change this to the desired value

# Use jq to update the value of "num_experts_per_tok" in the JSON file
# sed -i.bak "s/\"num_experts_per_tok\":.*/\"num_experts_per_tok\": $new_value,/" "$json_file_path"


# echo "Updated 'num_experts_per_tok' to $new_value in $json_file_path"
# #  
# # 
# /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu -i /home/stilex/dst/LLaMA-Factory/mixtral1/ncu_gsm8k_sparseb1_flash.ncu-rep > /home/stilex/gsm8k/gsm8k_dense/ncu_b1_flash.txt
# /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu --target-processes all --nvtx --nvtx-include "moe ffn/" --devices 0 -f -o ./mixtral1/ncu_gsm8k_sparseb1
# /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu -i /home/stilex/dst/LLaMA-Factory/mixtral1/ncu_b3dense.ncu-rep > /home/stilex/gsm8k/gsm8k_sparse/ncu_b3_flash.txt
# batch_sizes=(1 3)

# for batch_size in "${batch_sizes[@]}"; do
#     cp -f /home/stilex/copy_for_mamba/trainer.py /home/stilex/.local/lib/python3.8/site-packages/transformers
#     cp -f /home/stilex/copy_for_mamba/workflow.py /home/stilex/dst/LLaMA-Factory/src/llmtuner/train/sft
#     CUDA_VISIBLE_DEVICES=1  python src/train_bash.py \
#         --stage sft \
#         --do_train \
#         --dataset math \
#         --template mistral \
#         --finetuning_type lora \
#         --lora_target q_proj,v_proj \
#         --model_name_or_path Zyphra/BlackMamba-2.8B \
#         --output_dir /home/stilex/dst/LLaMA-Factory/moe_mamba \
#         --overwrite_output_dir \
#         --per_device_train_batch_size ${batch_size} \
#         --per_device_eval_batch_size 32 \
#         --gradient_accumulation_steps 1 \
#         --lr_scheduler_type cosine \
#         --logging_steps 10 \
#         --save_steps 600 \
#         --learning_rate 5e-5 \
#         --num_train_epochs 1 \
#         --cutoff_len 512 \
#         --bf16 > /home/stilex/dst/LLaMA-Factory/moe_mamba/math_throughput_b${batch_size}_dense.txt 
#     wait
# done  
#  /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu --target-processes all --nvtx --nvtx-include "moe ffn/" --devices 0 -f -o /home/stilex/dst/LLaMA-Factory/mixtral1/nmd
# nsys profile --gpu-metrics-device=1 --trace=cuda,nvtx,cudnn --stats=true -s cpu  --capture-range=cudaProfilerApi  --capture-range-end=stop --cuda-memory-usage true --output /home/stilex/nvtx/sparse_b9_204 --force-overwrite true
# CUDA_VISIBLE_DEVICES=1 python src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --dataset commonsense \
#     --template mistral_sum \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --model_name_or_path Zyphra/BlackMamba-2.8B \
#     --output_dir /home/stilex/dst/LLaMA-Factory/moe_mamba \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 1 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 1 \
#     --cutoff_len 1024 \
#     --bf16 > /home/stilex/profile_data/mamba/latency_breakdown/sparseb42_512.txt 

# /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu --target-processes all --replay-mode application --nvtx --nvtx-include "Backward" --devices 0 -f -o /data6/stilex/ncu_profile/mamba/sparseb3_512_back
json_file_path="../copy_for_mamba/trainer.py"
# List of batch sizes to iterate over
batch_sizes=(1 30 84)
seq_len=128
for batch_size in "${batch_sizes[@]}"; do
    # Update the BATCH_SIZE in the Python file
    sed -i.bak "s/^BATCH_SIZE = [0-9]\+$/BATCH_SIZE = $batch_size/" "$json_file_path"
    sed -i.bak "s/^SEQ_LEN = [0-9]\+$/SEQ_LEN = $seq_len/" "$json_file_path"

    cp -f ../copy_for_mamba/trainer.py /home/stilex/.local/lib/python3.8/site-packages/transformers
    
    # Run the training command with the updated batch size
    # -c 3000
    CUDA_VISIBLE_DEVICES=0  /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu  --target-processes all --nvtx  --nvtx-include "moe" --replay-mode application --devices 0 -f -o ./profile_data/blackmamba/ncu/sparseb${batch_size}_${seq_len} python src/train_bash.py \
        --stage sft \
        --do_train \
        --model_name_or_path Zyphra/BlackMamba-2.8B \
        --dataset math \
        --template mistral \
        --finetuning_type lora \
        --lora_target w1,w2,w3,gate \
        --output_dir Blackmamba \
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
        --bf16 > ./profile_data/blackmamba/ncu/sparseb${batch_size}_${seq_len}.txt

    /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu -i ./profile_data/blackmamba/ncu/sparseb${batch_size}_${seq_len}.ncu-rep > ./profile_data/blackmamba/ncu/sparseb${batch_size}_${seq_len}ncu.txt 
    # Optionally wait for the process to complete before starting the next iteration
    wait
done




new_value=8  # Change this to the desired value

sed -i.bak "s/\"topk\":.*/\"topk\": $new_value,/" "$config_file_path"


echo "Updated 'topk' to $new_value in $config_file_path"
cp -f $config_file_path $model_path


batch_sizes=(1 30)
seq_len=128
for batch_size in "${batch_sizes[@]}"; do
    # Update the BATCH_SIZE in the Python file
    sed -i.bak "s/^BATCH_SIZE = [0-9]\+$/BATCH_SIZE = $batch_size/" "$json_file_path"
    sed -i.bak "s/^SEQ_LEN = [0-9]\+$/SEQ_LEN = $seq_len/" "$json_file_path"

    cp -f ../copy_for_mamba/trainer.py /home/stilex/.local/lib/python3.8/site-packages/transformers
    
    # Run the training command with the updated batch size
    # -c 3000
    CUDA_VISIBLE_DEVICES=0  /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu  --target-processes all --nvtx  --nvtx-include "moe" --replay-mode application --devices 0 -f -o ./profile_data/blackmamba/ncu/denseb${batch_size}_${seq_len} python src/train_bash.py \
        --stage sft \
        --do_train \
        --model_name_or_path Zyphra/BlackMamba-2.8B \
        --dataset math \
        --template mistral \
        --finetuning_type lora \
        --lora_target w1,w2,w3,gate \
        --output_dir Blackmamba \
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
        --bf16 > ./profile_data/blackmamba/ncu/denseb${batch_size}_${seq_len}.txt

    /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu -i ./profile_data/blackmamba/ncu/denseb${batch_size}_${seq_len}.ncu-rep > ./profile_data/blackmamba/ncu/denseb${batch_size}_${seq_len}ncu.txt 
    # Optionally wait for the process to complete before starting the next iteration
    wait
done
# CUDA_VISIBLE_DEVICES=0 /usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu  --target-processes all --nvtx  --nvtx-include "moe" --replay-mode application --devices 0 -f -o /home/stilex/profile_data/mamba/ncu/sparseb1_512 python src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --dataset xsum \
#     --template mistral_sum \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --model_name_or_path Zyphra/BlackMamba-2.8B \
#     --output_dir /home/stilex/dst/LLaMA-Factory/moe_mamba \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 1 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 1 \
#     --cutoff_len 1024 \
#     --bf16 > /home/stilex/profile_data/mamba/ncu/sparseb1_512.txt &
    #      
    #  > /home/stilex/gsm8k/gsm8k_membreak/sparseb1.txt
# nsys profile --gpu-metrics-device=0 --trace=cuda,nvtx,cudnn --stats=true -s cpu  --capture-range=cudaProfilerApi  --capture-range-end=stop --cuda-memory-usage true --output /home/stilex/nvtx/dense_b1_204 --force-overwrite true
# CUDA_VISIBLE_DEVICES=2  nsys profile --gpu-metrics-device=2 --trace=cuda,nvtx  --cuda-memory-usage true --output /home/stilex/dst/LLaMA-Factory/sparse_batch6/sparse_b6_196 --force-overwrite true python src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
#     --dataset alpaca_en \
#     --template mistral \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir sparse_batch6 \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 6 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 500 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 1 \
#     --quantization_bit 4 \
#     --bf16  > /home/stilex/dst/LLaMA-Factory/profile_data/sparse_b6_196.log



# # Specify the new value for "num_experts_per_tok"
# new_value=8  # Change this to the desired value

# # Use jq to update the value of "num_experts_per_tok" in the JSON file
# sed -i.bak "s/\"num_experts_per_tok\":.*/\"num_experts_per_tok\": $new_value,/" "$json_file_path"

# echo "Updated 'num_experts_per_tok' to $new_value in $json_file_path"

# CUDA_VISIBLE_DEVICES=3  nsys profile --gpu-metrics-device=3 --trace=cuda,nvtx  --cuda-memory-usage true --output /home/stilex/dst/LLaMA-Factory/dense_batch1/dense_b1_196 --force-overwrite true python src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
#     --dataset alpaca_en \
#     --template mistral \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir  dense_batch1 \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 500 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 1 \
#     --quantization_bit 4 \
#     --bf16  > /home/stilex/dst/LLaMA-Factory/profile_data/dense_b1_196.log
    # --val_size 0.01 
# nsys profile --gpu-metrics-device=0 --trace=cuda,nvtx --export sqlite --output /home/stilex/dst/LLaMA-Factory/nsight_mixtral8x7b/mixtral8x7b_num2_b4 --force-overwrite true  
    # CUDA_VISIBLE_DEVICES=2 nsys profile --trace=cuda,nvtx --output /home/stilex/model_test/test --force-overwrite true test_compressed 
#  --export sqlite
#   
#!/bin/bash

# Specify the path to the JSON file


#!/bin/bash

# # Specify the path to the JSON file
# json_file_path="/data3/llama_model/yuchen/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841/config.json"

# # Specify the new value for "num_experts_per_tok"
# new_value=2  # Change this to the desired value

# # Use sed to update the value of "num_experts_per_tok" in the JSON file
# sed -i.bak "s/\"num_experts_per_tok\":.*/\"num_experts_per_tok\": $new_value,/" "$json_file_path"

# echo "Updated 'num_experts_per_tok' to $new_value in $json_file_path"

# # Function to run a Python script in the background
# run_python_script() {
#     CUDA_VISIBLE_DEVICES=$1 nsys profile --gpu-metrics-device=$1 --trace=cuda,nvtx --cuda-memory-usage true --output "$2" --force-overwrite true python src/train_bash.py \
#         --stage sft \
#         --do_train \
#         --model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
#         --dataset alpaca_en \
#         --template mistral \
#         --finetuning_type lora \
#         --lora_target q_proj,v_proj \
#         --output_dir "$3" \
#         --overwrite_output_dir \
#         --per_device_train_batch_size $4 \
#         --per_device_eval_batch_size $5 \
#         --gradient_accumulation_steps $6 \
#         --lr_scheduler_type cosine \
#         --logging_steps 10 \
#         --save_steps 500 \
#         --learning_rate 5e-5 \
#         --num_train_epochs 1 \
#         --quantization_bit 4 \
#         --bf16 > "$7" &
# }

# # Run Python scripts in parallel
# run_python_script 1 "/home/stilex/dst/LLaMA-Factory/sparse_batch1/sparse_b1_196" "sparse_batch1" 1 1 1 "/home/stilex/dst/LLaMA-Factory/profile_data/sparse_b1_196.log"
# run_python_script 2 "/home/stilex/dst/LLaMA-Factory/sparse_batch6/sparse_b6_196" "sparse_batch6" 6 1 1 "/home/stilex/dst/LLaMA-Factory/profile_data/sparse_b6_196.log"
# wait
# new_value=8  # Change this to the desired value

# # Use sed to update the value of "num_experts_per_tok" in the JSON file
# sed -i.bak "s/\"num_experts_per_tok\":.*/\"num_experts_per_tok\": $new_value,/" "$json_file_path"

# echo "Updated 'num_experts_per_tok' to $new_value in $json_file_path"
# wait
# run_python_script 3 "/home/stilex/dst/LLaMA-Factory/dense_batch1/dense_b1_196" "dense_batch1" 1 1 1 "/home/stilex/dst/LLaMA-Factory/profile_data/dense_b1_196.log"

# # Wait for all background processes to finish
# wait
