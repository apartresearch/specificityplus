#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')   

model = "gpt2-medium"
hparams_fname = "gpt2-medium.json"
alg_name = "ROME"
filename = "exp_gpt2medium.txt"

file_path = "experiment-scripts/" + filename

examples = 22000
split_into = 2

dataset_size = examples // split_into
if dataset_size * split_into != examples:
    raise ValueError("Dataset size must be divisible by split_into")

base_call = (f"cd git/memitpp && python experiments/evaluate.py --alg_name {alg_name} --model_name {model} \
--hparams_fname {hparams_fname} --ds_name cf --verbose --dataset_size {dataset_size}")

#get starting indexes
start_indexes = [i * examples // split_into for i in range(split_into)]

nr_expts = split_into

print(f'Total experiments = {nr_expts}')

output_file = open(file_path, "w")

for start_i in start_indexes:  
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--start_index {start_i} "
    )
    print(expt_call, file=output_file)

output_file.close()