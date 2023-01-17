
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

model_name ="EleutherAI/gpt-neox-20b"
config = AutoConfig.from_pretrained(model_name)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
print(model)