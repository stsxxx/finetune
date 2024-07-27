# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from huggingface_hub import login
from mamba_model import MambaModel
login()
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
model = MambaModel.from_pretrained(first_download=True,pretrained_model_name="Zyphra/BlackMamba-2.8B")