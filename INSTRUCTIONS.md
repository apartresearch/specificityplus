Step 0: download git and setup enviroment
1) git clone this repo https://github.com/jas-ho/memitpp
2) Setup conda enviroment: conda env create -f scripts/environment.yml
3) conda activate memit

Step 1: downoad models + data
1) switch to experiments branch
2) python setup_data/download_data.py
3) python setup_data/download_hfdata.py
4) python setup_data/download_models.py

Step 2: precompute layer stats
1) switch to collect_layers_stats branch
1) rome/layer_stats.py --model_name gpt2-medium --layers 8 --to_collect mom2 --precision float32 --download 1

Step 3: Run experiments
1) switch to experiments branch
2) change PATH_TO_REPO variable (to whatever your path is) in experiment-scripts/generate_experiment_file.py
3) python experiment-scripts/generate_experiment_file.py 
4) can edit (to run less parallel) by changing "split_into" variable in experiment-scripts/generate_experiment_file.py, and running the file to generate a new experiment txt
5) find lines to run in experiment-scripts/exp_gpt2medium.txt
6) Results for gpt2-medium are stored in results/


IF everything works for gpt2-medium, can run gpt2-xl, gpt-J, gpt-neox
1) go into setup_data/download_models.py and add the models to the "models" variable
2) python setup_data/download_models.py
3) switch to collect_layers_stats branch
4) python rome/layer_stats.py --model_name gpt2-xl --layers 17 --to_collect mom2 --precision float32 --download 1
5) switch to experiments branch
6) find lines to run in experiment-scripts/exp_gpt2xl.txt (can modify like previously if needed)
7) Results for gpt2-xl are stored in results/
8) switch to collect_layers_stats branch
9) cd git/memitpp && python rome/layer_stats.py --model_name EleutherAI/gpt-j-6B --layers 5 --to_collect mom2 --precision float32 --download 1
10) switch to experiments branch
11) find lines to run in experiment-scripts/exp_gptJ6B.txt (can modify like previously if needed)
12) Results for gpt-J are stored in results/
13) switch to collect_layers_stats branch
14) cd git/memitpp && python rome/layer_stats.py --model_name EleutherAI/gpt-neox-20b --layers 15 --to_collect mom2 --precision float32 --download 1
15) switch to experiments branch
16) find lines to run in experiment-scripts/exp_gptneox20b.txt (can modify like previously if needed)
17) Results for gpt-NEOX are stored in results/