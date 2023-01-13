## run on headnode

from transformers import AutoModelForCausalLM, AutoTokenizer

#get models
models = ["gpt2-xl", "EleutherAI/gpt-j-6B"]
for model_name in models:
    AutoModelForCausalLM.from_pretrained(model_name,resume_download = True)
    AutoTokenizer.from_pretrained(model_name,resume_download = True)