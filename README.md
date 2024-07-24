## Software Requirement

- Python 3.8
- CUDA 11.8
- CUDA toolkit 11.8
- torch==2.1.0+cu118
- transformers==4.40.0
- datasets==2.18.0
- accelerate==0.25.0
- peft==0.7.1
- trl==0.8.6
- bitsandbytes==0.42.0
- causal-conv1d==1.1.1
- mamba-ssm==1.2.0.post1
- tokenizers==0.19.1
- flash-attn==2.5.6
- huggingface-hub==0.20.3
- packaging==23.2

## Hardware Requirement

Experiments are conducted using NVIDIA A40 GPU with 48GB memory.

## Getting Started

First, install all required libraries

```bash
# install PyTorch 2.1.0 compatible with CUDA 11.8
pip install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# install other dependencies
pip install -r requirements.txt
```

Get into the LLaMa-Factory directory

```bash
cd LLaMA-Factory
```

Then download Mixtral and BlackMamba models from huggingface

```bash
#specify where you want to store models
export HF_HOME="path"
#download models
python3 model_download.py
```

Please change the transformers library path and model config file path in mamba_lt.sh, mamba_pf.sh, mamba_tp.sh, mixtral_lt.sh, mixtral_pf.sh and mixtral_tp.sh before running each bash script.


```bash
# change it to your transformers library path i.e. "/home/xxx/.local/lib/python3.8/site-packages/transformers"
transformers_path="xxxxx"
# change it to the huggingface hub path where the model config is stored i.e. "/xxxx/hub models--mistralai--Mixtral-8x7B-v0.1/snapshots/521a77772f0d4052fd9846846471d0d2517739d2"
config_file_path="xxxxx"
```


## Experiments Workflow


## Throughput 
You can reproduce the results for Figure 8 in the paper by running:
```bash
./mixtral_tp.sh
python3 throughput.py ./profile_data/mixtral/throughput > mixtral_throughput.txt
```
```bash
./mamba_tp.sh
python3 throughput.py ./profile_data/blackmamba/throughput > mamba_throughput.txt
```


## High-level and layer-level Latency Breakdown and Token Distribution

You can reproduce the results for Figure 4 and 5 in the paper by running:
```bash
./mixtral_lt.sh
python3 mixtral_latency.py ./profile_data/mixtral/latency > mixtral_latency_breakdown.txt
```
```bash
./mamba_lt.sh
python3 mamba_latency.py ./profile_data/blackmamba/latency > mamba_latency_breakdown.txt
```


## Kernel-Level latency breakown, SM and MEM Utilization

You can reproduce the results for Figure 6, 8 and 9 in the paper by running:
```bash
./mixtral_pf.sh
python3 sm_mixtral.py ./profile_data/mixtral/ncu > mixtral_sm.txt
python3 mem_mixtral.py ./profile_data/mixtral/ncu > mixtral_mem.txt
```
```bash
./mamba_pf.sh
python3 sm_mamba.py ./profile_data/blackmamba/ncu > mamba_sm.txt
python3 mem_mamba.py ./profile_data/blackmamba/ncu > mamba_mem.txt
```

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@inproceedings{xia2024Understanding,
  title={Understanding The Performance and Estimating The Cost Of LLM Fine-Tuning},
  author={Yuchen Xia and Jiho Kim and Yuhan Chen and Haojie Ye and Souvik Kundu and Cong "Callie" Hao and Nishil Talati},
  booktitle={Proceedings of 2024 IEEE International Symposium on Workload Characterization},
  address={Vancouver, Canada},
  year={2024},
}
```