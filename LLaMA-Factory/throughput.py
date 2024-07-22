import os
import re
import argparse

def read_log_file(file_path):
    with open(file_path, 'r') as file:
            for line in file:
                if "'train_runtime':" in line:
                    # Extract the value after the colon and strip any extra spaces or commas
                    runtime_value = line.split(":")[1].split(",")[0]
                    # print(runtime_value)
                    return float(runtime_value)
    print("Error: no training time!")
    return float(1)


def process_directory(directory_path):
    tp = 0
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            print(f"File: {file_name}")
            file_path = os.path.join(directory_path, file_name)
            tune_time = read_log_file(file_path)
            tp = 1000 / tune_time
            print(f"Throughput: {tp} queries/second")
            




            


            print()

def main():
    parser = argparse.ArgumentParser(description='Calculate average times for each log file in a directory.')
    parser.add_argument('directory', type=str, help='Path to the directory containing log files')

    args = parser.parse_args()
    log_directory = args.directory

    process_directory(log_directory)

if __name__ == "__main__":
    main()