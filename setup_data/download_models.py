## run on headnode

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

#get models
models = ["gpt2-xl", "gpt2-medium", "/sgugger/sharded-gpt-j-6B"]

for model_name in models:
    hf_hub_download(model_name,"pytorch_model.bin",resume_download = True)
    AutoTokenizer.from_pretrained(model_name,resume_download = True)