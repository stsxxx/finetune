# Open the file and read the data
import os
import re
import argparse
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
            unit = line.split()[1]
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
    silu_dur = []
    silu_ut = []
    silu_back_dur = []
    silu_back_ut = []
    element_mult_dur = []
    element_mult_sm = []
    w1w2_du = []
    w1w2_ut = []
    w3_du = []
    w3_ut = []
    gemm_du = []
    gemm_ut = []
    dequan_dur = []
    dequan_sm = []
    topk_dur = []
    topk_ut = []
    soft_dur = []
    soft_ut = []
    soft_back_dur = []
    soft_back_ut = []
    count = 0
    c = 0
    # print(len(name))
    moe_start_idx = 0
    moe_end_idx = 0
    atten_start_idx = 0
    atten_end_idx = 0
    for i in range(len(durations)):
        if 'gemm' in name[i]:
            if c == 204:
                moe_end_idx = i
            if c == 208:
                atten_end_idx = i
            c += 1
            gemm_du.append(durations[i])
            gemm_ut.append(compute_sms[i])
            # if count % 3 == 2:
            #     w3_du.append(durations[i])
            #     w3_ut.append(compute_sms[i])
            # else:
            #     w1w2_du.append(durations[i])
            #     w1w2_ut.append(compute_sms[i])
            # count += 1
        elif 'silu_kernel' in name[i]:
            silu_dur.append(durations[i])
            silu_ut.append(compute_sms[i])
        elif 'silu_backward_kernel' in name[i]:
            silu_back_dur.append(durations[i])
            silu_back_ut.append(compute_sms[i])
        elif 'gatherTopK' in name[i]:
            topk_dur.append(durations[i])
            topk_ut.append(compute_sms[i])
        elif 'softmax_warp_forward' in name[i]:
            soft_dur.append(durations[i])
            soft_ut.append(compute_sms[i])
        elif 'softmax_warp_backward' in name[i]:
            soft_back_dur.append(durations[i])
            soft_back_ut.append(compute_sms[i])
        elif 'BinaryFunctor' in name[i]:
            element_mult_dur.append(durations[i])
            element_mult_sm.append(compute_sms[i])
        elif 'kDequantize' in name[i]:
            if count == 58:
                moe_start_idx = i
            if count == 108:
                atten_start_idx = i
            count +=1
            dequan_dur.append(durations[i])
            dequan_sm.append(compute_sms[i]) 
    moe_dur = sum(durations[moe_start_idx:moe_end_idx+1])
    attn_dur = sum(durations[atten_start_idx: atten_end_idx+1])
    
        # avg_sm_util +=  compute_sms[i] * (durations[i] / sum_of_duration)
        # avg_mem_util +=  memory_sms[i] * (durations[i] / sum_of_duration)
    # print(len(w3_du))
    # print(len(w1w2_du))
    gemm_du = gemm_du[1:]
    gemm_ut = gemm_ut[1:]
    ngemm_du = [0] * 208
    ngemm_ut = [0] * 208
    dequan_dur = dequan_dur[1::2]
    dequan_sm = dequan_sm[1::2]
    # print(len(dequan_dur))
    ndequan_dur = [0] * 58
    ndequan_ut = [0] * 58

    # print(len(gemm_du))
    layer_num = len(gemm_du)//208
    # print(layer_num)
    for i in range(layer_num):
        for j in range(208):
            ngemm_du[j] += gemm_du[i * 208 + j]
            ngemm_ut[j] += gemm_ut[i * 208 + j]
    for i in range(layer_num):
        for j in range(58):
            ndequan_dur[j] += dequan_dur[i*58 + j]
            ndequan_ut[j] += dequan_sm[i*58 + j]

    ndequan_dur = [x / layer_num for x in ndequan_dur][4:-4]
    ndequan_ut = [x / layer_num for x in ndequan_ut][4:-4]

    for i in range(int(len(ndequan_dur)/2)):
        ndequan_dur[i] = (ndequan_dur[i] + ndequan_dur[len(ndequan_dur)-1-i]) /2
        ndequan_ut[i] = (ndequan_ut[i] + ndequan_ut[len(ndequan_ut)-1-i]) /2
    ndequan_dur = ndequan_dur[0:25]
    ndequan_ut = ndequan_ut[0:25]
    router_dequan_dur = ndequan_dur[0]
    w1_dequan_dur = ndequan_dur[1::3]
    w3_dequan_dur = ndequan_dur[2::3]
    w2_dequan_dur = ndequan_dur[3::3]
    router_dequan_ut = ndequan_ut[0]
    w1_dequan_ut = ndequan_ut[1::3]
    w3_dequan_ut = ndequan_ut[2::3]
    w2_dequan_ut = ndequan_ut[3::3]
    # print(ndequan_ut)
    ngemm_du = [x / layer_num for x in ngemm_du]
    ngemm_ut = [x / layer_num for x in ngemm_ut]
    ngemm_du = ngemm_du[4:-4]
    ngemm_ut = ngemm_ut[4:-4]
    # router_dequan_for_dur = ndequan_dur[0]
    # router_dequan_for_ut = ndequan_ut[0]
    # router_dequan_back_dur = ndequan_dur[-1]
    # router_dequan_back_ut = ndequan_ut[-1]

    # print(len(ndequan_dur))
    # print(len(ngemm_du))
    router_gemm_for_dur = ngemm_du[0:3][0]
    router_gemm_for_ut = ngemm_ut[0:3][0]
    # print(router_gemm_for_dur)

    router_gemm_back_dur = ngemm_du[-5:][-1]
    router_gemm_back_ut = ngemm_ut[-5:][-1]
    # print(router_gemm_back_dur)

    moe_gemm_for_dur = ngemm_du[3:75][0::3]
    moe_gemm_for_ut = ngemm_ut[3:75][0::3]
    # print(moe_gemm_for_dur)
    moe_gemm_back_dur = ngemm_du[75:195][4::5]
    moe_gemm_back_ut = ngemm_ut[75:195][4::5]
    # print(moe_gemm_back_dur)
    w1_for_dur = moe_gemm_for_dur[0::3]
    w1_for_ut = moe_gemm_for_ut[0::3]
    w3_for_dur = moe_gemm_for_dur[1::3]
    w3_for_ut = moe_gemm_for_ut[1::3]
    w2_for_dur = moe_gemm_for_dur[2::3]
    w2_for_ut = moe_gemm_for_ut[2::3]
    w2_back_dur = moe_gemm_back_dur[0::3]
    w2_back_ut = moe_gemm_back_ut[0::3]
    w3_back_dur = moe_gemm_back_dur[1::3]
    w3_back_ut = moe_gemm_back_ut[1::3]
    w1_back_dur = moe_gemm_back_dur[2::3]
    w1_back_ut = moe_gemm_back_ut[2::3]

    # print(w1_back_dur)

    # print(len(element_mult_dur))
    # element_mult_dur = element_mult_dur[::2]
    # element_mult_sm = element_mult_sm[::2]
    # dequan_dur = dequan_dur[1::2]
    # dequan_sm = dequan_sm[1::2]
    # print(len(element_mult_dur))
    avg_w3_for_util = 0
    avg_w1_for_util = 0
    avg_w2_for_util = 0
    avg_w3_back_util = 0
    avg_w1_back_util = 0
    avg_w2_back_util = 0
    avg_silu_for_util = 0 
    avg_silu_back_util = 0
    avg_topk_util = 0
    avg_element_util = 0
    avg_dquan_util = 0
    avg_soft_for_util = 0
    avg_soft_back_util = 0
    avg_w1_dequan_util = 0
    avg_w2_dequan_util = 0
    avg_w3_dequan_util = 0

    avg_w3_for_du = sum(w3_for_dur) /len(w3_for_dur)
    avg_w2_for_du = sum(w2_for_dur) /len(w2_for_dur)
    avg_w1_for_du = sum(w1_for_dur) /len(w1_for_dur)
    avg_w3_back_du = sum(w3_back_dur) /len(w3_back_dur)
    avg_w2_back_du = sum(w2_back_dur) /len(w2_back_dur)
    avg_w1_back_du = sum(w1_back_dur) /len(w1_back_dur)
    avg_topk_du = sum(topk_dur) /len(topk_dur)
    # avg_w1w2_du = sum(w1w2_du)/ len(w1w2_du)
    avg_silu_for_du = sum(silu_dur) / len(silu_dur)
    avg_silu_back_du = sum(silu_back_dur) / len(silu_back_dur)
    avg_soft_for_du = sum(soft_dur) / len(soft_dur)
    avg_soft_back_du = sum(soft_back_dur) / len(soft_back_dur)
    avg_elementmult_du = sum(element_mult_dur) / len(element_mult_dur)
    # avg_dquan_du = sum(dequan_dur) / len(dequan_dur)
    avg_w1_dequan_du = sum(w1_dequan_dur) / len(w1_dequan_dur)
    avg_w2_dequan_du = sum(w2_dequan_dur) / len(w2_dequan_dur)
    avg_w3_dequan_du = sum(w3_dequan_dur) / len(w3_dequan_dur)

    for i in range(len(w3_for_dur)):
        avg_w3_for_util +=  w3_for_ut[i] * (w3_for_dur[i] / sum(w3_for_dur))
    for i in range(len(w2_for_dur)):
        avg_w2_for_util +=  w2_for_ut[i] * (w2_for_dur[i] / sum(w2_for_dur))   
    for i in range(len(w1_for_dur)):
        avg_w1_for_util +=  w1_for_ut[i] * (w1_for_dur[i] / sum(w1_for_dur))
    for i in range(len(w3_back_dur)):
        avg_w3_back_util +=  w3_back_ut[i] * (w3_back_dur[i] / sum(w3_back_dur))
    for i in range(len(w2_back_dur)):
        avg_w2_back_util +=  w2_back_ut[i] * (w2_back_dur[i] / sum(w2_back_dur))   
    for i in range(len(w1_back_dur)):
        avg_w1_back_util +=  w1_back_ut[i] * (w1_back_dur[i] / sum(w1_back_dur))
    for i in range(len(topk_dur)):
        avg_topk_util +=  topk_ut[i] * (topk_dur[i] / sum(topk_dur))
        
        

    for i in range(len(silu_dur)):
        avg_silu_for_util +=  silu_ut[i] * (silu_dur[i] / sum(silu_dur))
    for i in range(len(silu_back_dur)):
        avg_silu_back_util +=  silu_back_ut[i] * (silu_back_dur[i] / sum(silu_back_dur))

    for i in range(len(soft_dur)):
        avg_soft_for_util +=  soft_ut[i] * (soft_dur[i] / sum(soft_dur))
    for i in range(len(soft_back_dur)):
        avg_soft_back_util +=  soft_back_ut[i] * (soft_back_dur[i] / sum(soft_back_dur))
        
    for i in range(len(element_mult_dur)):
        avg_element_util +=  element_mult_sm[i] * (element_mult_dur[i] / sum(element_mult_dur))
    for i in range(len(w1_dequan_ut)):
        avg_w1_dequan_util += w1_dequan_ut[i] * (w1_dequan_dur[i] / sum(w1_dequan_dur))
    for i in range(len(w2_dequan_ut)):
        avg_w2_dequan_util += w2_dequan_ut[i] * (w2_dequan_dur[i] / sum(w2_dequan_dur))
    for i in range(len(w3_dequan_ut)):
        avg_w3_dequan_util += w3_dequan_ut[i] * (w3_dequan_dur[i] / sum(w3_dequan_dur))
    print('avg w2 forward duration:',avg_w2_for_du)
    print('avg w2 dequan duration:', avg_w2_dequan_du)
    print('avg w3 forward duration:',avg_w3_for_du)
    print('avg w3 dequan duration:', avg_w3_dequan_du)
    print('avg silu forward duration:',avg_silu_for_du)
    print('avg w1 forward duration:',avg_w1_for_du)
    print('avg w1 dequan duration:', avg_w1_dequan_du)
    print('avg softmax forward duration:',avg_soft_for_du)
    print('avg topk duration:', avg_topk_du)
    print('avg router gemm forward duration:', router_gemm_for_dur)
    print('avg router dequan duration:', router_dequan_dur)









    # print('avg router dequan forward duration:',avg_soft_for_du)

    print('avg w2 backward duration:',avg_w2_back_du)
    print('avg w2 dequan duration:', avg_w2_dequan_du)
    print('avg w3 backward duration:',avg_w3_back_du)
    print('avg w3 dequan duration:', avg_w3_dequan_du)
    print('avg silu backward duration:',avg_silu_back_du)
    print('avg w1 backward duration:',avg_w1_back_du)
    print('avg w1 dequan duration:', avg_w1_dequan_du)
    print('avg softmax backward duration:',avg_soft_back_du)
    print('avg router gemm backward duration:', router_gemm_back_dur)
    print('avg router dequan duration:', router_dequan_dur)

    # print('avg router dequan backward duration:',avg_soft_back_du)


    print('avg w2 forward utilization:',avg_w2_for_util)
    print('avg w2 dequan utilization:', avg_w2_dequan_util)
    print('avg w3 forward utilization:',avg_w3_for_util)
    print('avg w3 dequan utilization:', avg_w3_dequan_util)
    print('avg silu forward utilization:',avg_silu_for_util)
    print('avg w1 forward utilization:',avg_w1_for_util)
    print('avg w1 dequan utilization:', avg_w1_dequan_util)
    print('avg softmax forward utilization:',avg_soft_for_util)
    print('avg topk utilization:', avg_topk_util)
    print('avg router gemm forward utilization:', router_gemm_for_ut)
    print('avg router dequan utilization:', router_dequan_ut)


    # print('avg softmax forward utilization:',avg_soft_for_util)
    print('avg w2 backward utilization:',avg_w2_back_util)
    print('avg w2 dequan utilization:', avg_w2_dequan_util)
    print('avg w3 backward utilization:',avg_w3_back_util)
    print('avg w3 dequan utilization:', avg_w3_dequan_util)
    print('avg silu backward utilization:',avg_silu_back_util)
    print('avg w1 backward utilization:',avg_w1_back_util)
    print('avg w1 dequan utilization:', avg_w1_dequan_util)
    print('avg softmax backward utilization:',avg_soft_back_util)
    print('avg router gemm backward utilization:', router_gemm_back_ut)
    print('avg router dequan utilization:', router_dequan_ut)

    # print('avg softmax backward utilization:',avg_soft_back_util)





    # print('avg elementwise duration:',avg_elementmult_du)
    # print('avg elementwise utilization:',avg_element_util)
    # print('avg dequant duration:', avg_dquan_du)
    # print('avg dequant utilization:', avg_dquan_util)
    # print('avg sm:', avg_sm_util)
    # print('avg mem:', avg_mem_util)
    print('avg moe time:', moe_dur* 32 / 1000000)
    print('avg atten time:', attn_dur* 32 / 1000000)
    total_avg_dur = (
        avg_w2_for_du + avg_w2_dequan_du + avg_w3_for_du + avg_w3_dequan_du + 
        avg_silu_for_du + avg_w1_for_du + avg_w1_dequan_du + avg_soft_for_du + 
        avg_topk_du + router_gemm_for_dur + router_dequan_dur +
        avg_w2_back_du + avg_w3_back_du + avg_silu_back_du + avg_w1_back_du + 
        avg_soft_back_du + router_gemm_back_dur
    )
    weighted_util = (
        (avg_w2_for_util * avg_w2_for_du / total_avg_dur) +
        (avg_w2_dequan_util * avg_w2_dequan_du / total_avg_dur) +
        (avg_w3_for_util * avg_w3_for_du / total_avg_dur) +
        (avg_w3_dequan_util * avg_w3_dequan_du / total_avg_dur) +
        (avg_silu_for_util * avg_silu_for_du / total_avg_dur) +
        (avg_w1_for_util * avg_w1_for_du / total_avg_dur) +
        (avg_w1_dequan_util * avg_w1_dequan_du / total_avg_dur) +
        (avg_soft_for_util * avg_soft_for_du / total_avg_dur) +
        (avg_topk_util * avg_topk_du / total_avg_dur) +
        (router_gemm_for_ut * router_gemm_for_dur / total_avg_dur) +
        (router_dequan_ut * router_dequan_dur / total_avg_dur) +
        (avg_w2_back_util * avg_w2_back_du / total_avg_dur) +
        (avg_w3_back_util * avg_w3_back_du / total_avg_dur) +
        (avg_silu_back_util * avg_silu_back_du / total_avg_dur) +
        (avg_w1_back_util * avg_w1_back_du / total_avg_dur) +
        (avg_soft_back_util * avg_soft_back_du / total_avg_dur) +
        (router_gemm_back_ut * router_gemm_back_dur / total_avg_dur)
    )
    print('weighted utilization:', weighted_util)
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