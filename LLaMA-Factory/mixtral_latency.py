import os
import re
import argparse

def read_log_file(file_path):
    self_attention_times = []
    moe_times = []
    input_norm_times = []
    after_attention_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    moe_ffc_times = []
    moe_mask_times = []
    token_nums = {} 
    epoch_time = []
    train_step_time = []
    attention_mem = []
    moe_mem = []
    inputnorm_mem =[]
    outputnorm_mem = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            self_attention_match = re.search(r'self-attention layer time: (\d+\.\d+)', line)
            moe_match = re.search(r'moe layer time: (\d+\.\d+)', line)
            input_norm_match = re.search(r'input normlization layer time: (\d+\.\d+)', line)
            after_norm_match = re.search(r'post attention normlization layer time: (\d+\.\d+)', line)
            forward_match = re.search(r'forward time: (\d+\.\d+)', line)
            backward_match = re.search(r'backward time: (\d+\.\d+)', line)
            op_match = re.search(r'optimizer iteration time: (\d+\.\d+)', line)
            moe_ffc_match = re.search(r'moe ffc time: (\d+\.\d+)', line)
            moe_mask_match = re.search(r'moe mask calculation time: (\d+\.\d+)', line)
            token_match = re.search(r'token num for expert (\d+): (\d+)', line)
            epoch_match = re.search(r'epoch time: (\d+\.\d+)', line)
            train_step_match = re.search(r'train step time: (\d+\.\d+)', line)
            attention_mem_match = re.search(r'Attention used (\d+\.\d+) GB', line)
            moe_mem_match = re.search(r'MOE used (\d+\.\d+) GB', line)
            inputnorm_mem_match = re.search(r'input normlization used (\d+\.\d+) GB', line)
            outputnorm_mem_match = re.search(r'post attention normlization used (\d+\.\d+) GB', line)
            if token_match:
                expert_idx = int(token_match.group(1))
                token_num = float(token_match.group(2))
                
                if expert_idx not in token_nums:
                    token_nums[expert_idx] = []
                # Store token number for the corresponding expert index
                token_nums[expert_idx].append(token_num)
            if self_attention_match:
                self_attention_times.append(float(self_attention_match.group(1)))
            if forward_match:
                forward_times.append(float(forward_match.group(1)))
            if backward_match:
                backward_times.append(float(backward_match.group(1)))
            if moe_match:
                moe_times.append(float(moe_match.group(1)))
            if input_norm_match:
                input_norm_times.append(float(input_norm_match.group(1)))
            if after_norm_match:
                after_attention_times.append(float(after_norm_match.group(1)))
            if op_match:
                optimizer_times.append(float(op_match.group(1)))
            if epoch_match:
                epoch_time.append(float(epoch_match.group(1)))   
            if train_step_match:
                train_step_time.append(float(train_step_match.group(1)))     
            if moe_ffc_match:
                moe_ffc_times.append(float(moe_ffc_match.group(1)))
            if moe_mask_match:
                moe_mask_times.append(float(moe_mask_match.group(1)))    
            if attention_mem_match:
                attention_mem.append(float(attention_mem_match.group(1)))
            if moe_mem_match:
                moe_mem.append(float(moe_mem_match.group(1)))
            if inputnorm_mem_match:
                inputnorm_mem.append(float(inputnorm_mem_match.group(1)))
            if outputnorm_mem_match:
                outputnorm_mem.append(float(outputnorm_mem_match.group(1)))

    return self_attention_times, moe_times, input_norm_times, after_attention_times, forward_times, backward_times,optimizer_times,moe_ffc_times,moe_mask_times, token_nums, epoch_time, train_step_time, attention_mem, moe_mem, inputnorm_mem, outputnorm_mem

def calculate_average(times):
    # print(sum(times))
    return (sum(times)) / len(times) if times else 0.0

def calculate_average_chunked(data, chunk_size):
    """
    Calculate average of data in chunks.
    """
    chunked_data = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    averages = [sum(chunk) / len(chunk) for chunk in chunked_data]
    forward_avg = 0
    backward_avg = 0
    print(len(averages))
    for i in range(len(averages)):
        if (i + 1) % 2 == 1:
            forward_avg += averages[i]
        else:
            backward_avg += averages[i]
    forward_avg = forward_avg / len(averages) * 2
    backward_avg = backward_avg / len(averages) * 2

    return forward_avg, backward_avg

def process_directory(directory_path):
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            self_attention_times, moe_times, input_norm_times, after_attention_times, forward_times, backward_times, optimizer_times,moe_ffc_times,moe_mask_times, token_nums,epoch_time, train_step_time, attention_mem, moe_mem, inputnorm_mem, outputnorm_mem = read_log_file(file_path)

            avg_self_attention_time_f,  avg_self_attention_time_b= calculate_average_chunked(self_attention_times,32)
            avg_moe_time_f, avg_moe_time_b = calculate_average_chunked(moe_times,32)
            avg_moe_ffc_time_f, avg_moe_ffc_time_b = calculate_average_chunked(moe_ffc_times,32)
            avg_moe_mask_time_f, avg_moe_mask_time_b = calculate_average_chunked(moe_mask_times,32)
            avg_input_norm_time_f,  avg_input_norm_time_b= calculate_average_chunked(input_norm_times,32)
            avg_output_norm_time_f, avg_output_norm_time_b= calculate_average_chunked(after_attention_times,32)
            avg_forward_time = calculate_average(forward_times)
            avg_backward_time = calculate_average(backward_times)
            avg_op_time = calculate_average(optimizer_times)
            avg_trainstep_time = calculate_average(train_step_time)
            # avg_epoch_time = calculate_average(epoch_time)
            # avg_moe_mem = calculate_average(moe_mem)
            # avg_attention_mem_f, avg_attention_mem_b = calculate_average_chunked(attention_mem,32)
            # avg_inputnorm_mem_f, avg_inputnorm_mem_b = calculate_average_chunked(inputnorm_mem,32)
            # avg_outputnorm_mem_f, avg_outputnorm_mem_b = calculate_average_chunked(outputnorm_mem,32)

            for expert_idx, token_list in token_nums.items():
                if token_list:
                    avg_token_num = sum(token_list) / len(token_list)
                    print(f"Average token num for expert {expert_idx}: {avg_token_num / 4}")
                    # for i in range(0, len(token_list), 32):
                    #     print(token_list[i])
                else:
                    print(f"No tokens for expert {expert_idx}")
            print(f"File: {file_name}")
            print(f"Forward Average input normalization Time: {avg_input_norm_time_f * 32} seconds")
            print(f"Forward Average Self-Attention Time: {avg_self_attention_time_f* 32} seconds")
            print(f"Forward Average MoE layer Time: {avg_moe_time_f* 32} seconds")
            print(f"Forward Average moe fully connected layer Time: {avg_moe_ffc_time_f* 32} seconds")
            print(f"Forward Average moe masked layer Time: {avg_moe_mask_time_f* 32} seconds")
            print(f"Forward Average post attention normlization Time: {avg_output_norm_time_f* 32} seconds")
            print(f"Average forward Time: {avg_forward_time} seconds")
            print(f"Backward Average input normalization Time: {avg_input_norm_time_b} seconds")
            print(f"Backward Average Self-Attention Time: {avg_self_attention_time_b} seconds")
            print(f"Backward Average MoE layer Time: {avg_moe_time_b} seconds")
            print(f"Backward Average moe fully connected layer Time: {avg_moe_ffc_time_b} seconds")
            print(f"Backward Average moe masked layer Time: {avg_moe_mask_time_b} seconds")
            print(f"Backward Average post attention normlization Time: {avg_output_norm_time_b} seconds")
            print(f"Average backward Time: {avg_backward_time} seconds")
            print(f"Average optimizer Time: {avg_op_time} seconds")
            print(f"Average train step Time: {avg_trainstep_time} seconds")
            # print(f"epoch Time: {avg_epoch_time} seconds")

            # print(f"Average MOE mem: {avg_moe_mem * 32} GB")
            # print(f"Forward Average attention mem: {avg_attention_mem_f *32} GB")
            # print(f"Backward Average attention mem: {avg_attention_mem_b*32} GB")
            # print(f"Forward Average inputnorm mem: {avg_inputnorm_mem_f*32} GB")
            # print(f"Backward Average inputnorm mem: {avg_inputnorm_mem_b*32} GB")
            # print(f"Forward Average post attention normlization mem: {avg_outputnorm_mem_f*32} GB")
            # print(f"Backward Average post attention normlization mem: {avg_outputnorm_mem_b*32} GB")






            


            print()

def main():
    parser = argparse.ArgumentParser(description='Calculate average times for each log file in a directory.')
    parser.add_argument('directory', type=str, help='Path to the directory containing log files')

    args = parser.parse_args()
    log_directory = args.directory

    process_directory(log_directory)

if __name__ == "__main__":
    main()