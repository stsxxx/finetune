from mamba_model import MambaModel
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MambaModel.from_pretrained(pretrained_model_name="Zyphra/BlackMamba-2.8B")
model = model.to(device)
inputs = torch.tensor([1, 2]).to(device).long().unsqueeze(0)
out = model(inputs)
# print(out)