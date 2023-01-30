from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig

model_name = "EleutherAI/gpt-neox-20b"
config = AutoConfig.from_pretrained(model_name)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
print(model)
