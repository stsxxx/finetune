import argparse
import os
def select_custom_pattern(array, select_count, skip_count,start_idx):
    result = []
    i = start_idx
    while i < len(array):
        # Select elements
        result.extend(array[i:i + select_count])
        # Skip elements
        i += select_count + skip_count
    return result# Open the file and read the data

def read_log_file(file_path):
    with open(file_path, "r") as file:
        data = file.read()

    # Split the data into lines and remove empty lines
    lines = [line.strip() for line in data.split('\n') if line.strip()]

    # Initialize lists to store Duration and Compute (SM) [%] values
    durations = []
    compute_sms = []
    memory_sms = []
    name = []
    # Iterate through lines to find Duration and Compute (SM) [%]
    for line in lines:
        if "Stream" in line:
            name.append(line)
        if "Duration" in line:
            duration = float(line.split()[2])
            unit = line.split()[2]
            # print("Durations:", duration)
            if unit == 'msecond':
                durations.append(duration * 1000)
            else:
                durations.append(duration)
        elif "Compute (SM)" in line:
            compute_sm = float(line.split()[4])
            # print("Compute (SM) [%]:", compute_sm)
            compute_sms.append(compute_sm)
        elif "Memory [%]" in line:
            memory_sm = float(line.split()[3])
            memory_sms.append(memory_sm)
    sum_of_duration = sum(durations)
    # avg_sm_util = 0
    # avg_mem_util = 0
    gemm_dur = []
    gemm_ut = []
    gelu_dur = []
    gelu_ut = []
    element_mult_dur = []
    element_mult_sm = []
    w1w2_du = []
    w1w2_ut = []
    r_gemm_du = []
    r_gemm_ut = []
    dequan_dur = []
    dequan_sm = []
    sigmoid_dur = []
    sigmoid_sm = []
    topk_dur = []
    topk_sm = []
    norm_dur =[]
    norm_sm = []
    count = 0
    cc = 0
    moe_start_idx = 0
    moe_end_idx = 0
    mamba_start_idx = 0
    mamba_end_idx = 0
    # print(len(name))
    for i in range(len(durations)):
        # print(name[i])

        if 'gemm' in name[i]:
            if count == 2:
                moe_start_idx = i
                print(i)
            if count == 35:
                moe_end_idx = i
                print(i)

            if count == 44:
                mamba_end_idx  = i
                print(i)

            gemm_dur.append(durations[i])
            gemm_ut.append(compute_sms[i])
            count += 1
            # if count % 17 == 0:
            #     r_gemm_du.append(durations[i])
            #     r_gemm_ut.append(compute_sms[i])
            # else:
            #     w1w2_du.append(durations[i])
            #     w1w2_ut.append(compute_sms[i])
            # count += 1
        elif'causal_conv1d_fwd_kernel' in name[i]:
            if cc == 0:
                mamba_start_idx = i
            cc += 1
        elif '_layer_norm_bwd_kernel' in name[i]:
            norm_dur.append(durations[i])
            norm_sm.append(compute_sms[i])
        elif 'sigmoid' in name[i]:
            sigmoid_dur.append(durations[i])
            sigmoid_sm.append(compute_sms[i])
        elif 'Gelu' in name[i]:
            gelu_dur.append(durations[i])
            gelu_ut.append(compute_sms[i])
        elif 'MulFunc' in name[i]:
            element_mult_dur.append(durations[i])
            element_mult_sm.append(compute_sms[i])
        elif 'kDequantize' in name[i]:
            dequan_dur.append(durations[i])
            dequan_sm.append(compute_sms[i]) 
        elif "gatherTopK" in name[i]:
            topk_dur.append(durations[i])
            topk_sm.append(compute_sms[i]) 
    gemm_dur = gemm_dur[2:]
    gemm_ut = gemm_ut[2:]
    layer_num = len(gemm_dur) // 43
    ngemm_du = [0] * 43
    ngemm_ut = [0] * 43
    # print(layer_num)
    for i in range(layer_num):
        for j in range(43):
            ngemm_du[j] += gemm_dur[i * 43 + j]
            ngemm_ut[j] += gemm_ut[i * 43 + j]
    ngemm_du = [x / layer_num for x in ngemm_du]
    ngemm_ut = [x / layer_num for x in ngemm_ut]
    ngemm_du = ngemm_du[:-9]
    ngemm_ut = ngemm_ut[:-9]
    router_gemm_du = ngemm_du[-2:]
    router_gemm_ut = ngemm_ut[-2:]

    # print(len(ngemm_du))
    w2_dur = select_custom_pattern(ngemm_du[:-2],2,2,0)
    w2_ut = select_custom_pattern(ngemm_ut[:-2],2,2,0)
    w1_dur = select_custom_pattern(ngemm_du[:-2],2,2,2)
    w1_ut = select_custom_pattern(ngemm_ut[:-2],2,2,2)
    # print(w2_ut)
        # avg_sm_util +=  compute_sms[i] * (durations[i] / sum_of_duration)
        # avg_mem_util +=  memory_sms[i] * (durations[i] / sum_of_duration)
    # print(len(w3_du))
    # print(len(w1w2_du))
    # print(len(silu_dur))
    # print(len(element_mult_dur))
    element_mult_dur = element_mult_dur[1::2]
    element_mult_sm = element_mult_sm[1::2]
    # dequan_dur = dequan_dur[1::2]
    # dequan_sm = dequan_sm[1::2]
    # print(len(element_mult_dur))
    # w1_dur = w1w2_du[::2]
    # w1_sm = w1w2_ut[::2]
    # w2_dur = w1w2_du[1::2]
    # w2_sm = w1w2_ut[1::2]

    avg_r_gemm_util = 0
    avg_w1_util = 0
    avg_w2_util = 0
    avg_topk_util = 0
    avg_gelu_util = 0
    avg_element_util = 0
    avg_sig_util = 0
    avg_r_gemm_du = sum(router_gemm_du) /len(router_gemm_du)
    avg_w1_du = sum(w1_dur)/ len(w1_dur)
    avg_w2_du = sum(w2_dur)/ len(w2_dur)
    # avg_topk_du = sum(topk_dur)/ len(topk_dur)
    avg_gelu_du = sum(gelu_dur) / len(gelu_dur)
    avg_sig_du = sum(sigmoid_dur) / len(sigmoid_dur)

    avg_elementmult_du = sum(element_mult_dur) / len(element_mult_dur)
    # avg_dquan_du = sum(dequan_dur) / len(dequan_dur)
    for i in range(len(router_gemm_du)):
        avg_r_gemm_util +=  router_gemm_ut[i] * (router_gemm_du[i] / sum(router_gemm_du))
    for i in range(len(w1_dur)):
        avg_w1_util +=  w1_ut[i] * (w1_dur[i] / sum(w1_dur))
    for i in range(len(w2_dur)):
        avg_w2_util +=  w2_ut[i] * (w2_dur[i] / sum(w2_dur))
    # for i in range(len(topk_dur)):
    #     avg_topk_util +=  topk_sm[i] * (topk_dur[i] / sum(topk_dur))
    for i in range(len(gelu_dur)):
        avg_gelu_util +=  gelu_ut[i] * (gelu_dur[i] / sum(gelu_dur))
    for i in range(len(sigmoid_dur)):
        avg_sig_util +=  sigmoid_sm[i] * (sigmoid_dur[i] / sum(sigmoid_dur))
    for i in range(len(element_mult_dur)):
        avg_element_util +=  element_mult_sm[i] * (element_mult_dur[i] / sum(element_mult_dur))
    # for i in range(len(dequan_dur)):
    #     avg_dquan_util += dequan_sm[i] * (dequan_dur[i] / sum(dequan_dur))
    moe_dur = sum(durations[moe_start_idx:moe_end_idx+1])
    # print(durations[mamba_start_idx: mamba_end_idx+1])
    mamba_dur = sum(durations[mamba_start_idx: mamba_end_idx+1])
    layernorm_dur = sum(norm_dur[49:51])
    print('avg w1 backward duriation:',avg_w1_du)
    print('avg gelu backward duriation:',avg_gelu_du)
    print('avg w2 backward duriation:',avg_w2_du)
    print('avg elementwise duriation:',avg_elementmult_du)
    print('avg sigmoid backward duriation:',avg_sig_du)
    print('avg router_gemm backward duration:', avg_r_gemm_du)
    print('avg w1 backward utilization:',avg_w1_util)
    print('avg gelu backward utilization:',avg_gelu_util)
    print('avg w2 backward utilization:',avg_w2_util)
    print('avg elementwise utilization:',avg_element_util)
    print('avg sigmoid backward utilization:',avg_sig_util)
    print('avg router_gemm backward utilization:', avg_r_gemm_util)
    print('avg moe time:', moe_dur* 18 / 1000000)
    print('avg mamba time:', mamba_dur* 18 / 1000000)
    print('avg layernorm time:', layernorm_dur * 18 / 1000000)




    # print('avg top_k duration:', avg_topk_du)
    # print('avg top_k utilization:', avg_topk_util)
    # print('avg sm:', avg_sm_util)
    # print('avg mem:', avg_mem_util)

    # # Print the extracted lists
    # print("Durations:", durations)
    # print("Compute (SM) [%]:", compute_sms)

def process_directory(directory_path):
    for file_name in os.listdir(directory_path):
        if file_name.endswith('ncu.txt'):
            file_path = os.path.join(directory_path, file_name)

            print(f"File: {file_name}")

            read_log_file(file_path)
            print()

def main():
    parser = argparse.ArgumentParser(description='Calculate average times for each log file in a directory.')
    parser.add_argument('directory', type=str, help='Path to the directory containing log files')

    args = parser.parse_args()
    log_directory = args.directory

    process_directory(log_directory)

if __name__ == "__main__":
    main()