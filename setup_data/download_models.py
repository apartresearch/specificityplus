## run on headnode

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

#get models

models = ["gpt2-medium","gpt2-xl"]#["gpt2-xl", "gpt2-medium", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"]

for model_name in models:
    print("Downloading model: ", model_name)
    snapshot_download(model_name,resume_download = True, ignore_patterns =["*.msgpack","*.h5","*.ot"])
    print("Downloading tokenizer: ", model_name)
    AutoTokenizer.from_pretrained(model_name,resume_download = True)
