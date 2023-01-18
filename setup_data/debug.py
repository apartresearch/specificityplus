from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
#cache dir
cache_dir = "/disk/scratch/s1785649/memitpp/data/huggingface/hub"
print("checkpoint 1")
config = AutoConfig.from_pretrained(cache_dir+"/"+"gpt-neox-20b/config.json")
print("checkpoint 2")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
print("checkpoint 3")
model = load_checkpoint_and_dispatch(
    model, cache_dir+"/"+"gpt-neox-20b", device_map="auto", no_split_module_classes=["GPTNeoXLayer"], dtype=torch.float16
)
print("works!!")

#print n_positions from config
print(model.config.n_positions)