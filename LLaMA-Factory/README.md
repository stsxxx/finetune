## Requirement

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers, Datasets, Accelerate, PEFT and TRL
- sentencepiece, protobuf and tiktoken
- jieba, rouge-chinese and nltk (used at evaluation and predict)
- gradio and matplotlib (used in web UI)
- uvicorn, fastapi and sse-starlette (used in API)

### Hardware Requirement

| Method | Bits |   7B  |  13B  |  30B  |   65B  |   8x7B |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ |
| Full   |  16  | 160GB | 320GB | 600GB | 1200GB |  900GB |
| Freeze |  16  |  20GB |  40GB | 120GB |  240GB |  200GB |
| LoRA   |  16  |  16GB |  32GB |  80GB |  160GB |  120GB |
| QLoRA  |   8  |  10GB |  16GB |  40GB |   80GB |   80GB |
| QLoRA  |   4  |   6GB |  12GB |  24GB |   48GB |   32GB |

## Getting Started


Please change the transformers library path and model config file path before running each bash script.


```bash
# change it to your transformers library path i.e. /home/xxx/.local/lib/python3.8/site-packages/transformers
transformers_path="xxxxx"
# change it to the huggingface hub path where the model config is stored i.e. "/xxxx/hub models--mistralai--Mixtral-8x7B-v0.1/snapshots/521a77772f0d4052fd9846846471d0d2517739d2"
config_file_path="xxxxx"
```

