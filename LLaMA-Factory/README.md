## Software Requirement

- Python 3.8
- Cuda 11.8
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

### Hardware Requirement

Experiments are conducted using NVIDIA A40 GPU with 48GB memory.

## Getting Started

First, download Mixtral and BlackMamba models from huggingface

```bash
python3 model_download.py
```

Please change the transformers library path and model config file path before running each bash script.


```bash
# change it to your transformers library path i.e. "/home/xxx/.local/lib/python3.8/site-packages/transformers"
transformers_path="xxxxx"
# change it to the huggingface hub path where the model config is stored i.e. "/xxxx/hub models--mistralai--Mixtral-8x7B-v0.1/snapshots/521a77772f0d4052fd9846846471d0d2517739d2"
config_file_path="xxxxx"
```


## Experiments Workflow


## Throughput 

```bash
./mixtral_tp.sh
python3 throughput.py ./profile_data/mixtral/throughput > mixtral_throughput.txt
```
```bash
./mamba_tp.sh
python3 throughput.py ./profile_data/blackmamba/throughput > mamba_throughput.txt
```


## High-level and layer-level Latency Breakdown and Token Distribution

```bash
./mixtral_lt.sh
python3 mixtral_latency.py ./profile_data/mixtral/latency > mixtral_latency_breakdown.txt
```
```bash
./mamba_lt.sh
python3 mamba_latency.py ./profile_data/blackmamba/latency > mamba_latency_breakdown.txt
```


## Kernel-Level latency breakown, SM and MEM Utilization

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