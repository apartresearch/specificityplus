from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").eval().cuda()
print(model)