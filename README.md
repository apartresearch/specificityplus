
# Specificity Plus

👩‍💻 Official Website for "Uncovering Limits of Memory Editing in Large Language Models: A New Specificity Benchmark" Aceepted at Findings of ACL 2023.

## [Visit the web page](https://specificityplus.apartresearch.com)
=======
# Detecting Edit Failures in LLMs: An Improved Specificity Benchmark

This repository contains the code for the paper [Detecting Edit Failures in LLMs: An Improved Specificity Benchmark]() (ACL Findings 2023). # todo: add link

It extends previous work on model editing by Meng et al. #todo: add citations by introducing a new benchmark, called CounterFact+, for measuring the specificity of model edits. 

### Attribution
The repository is a fork of https://github.com/kmeng01/memit, which implement the model editing algorithms MEMIT (Mass Editing Memory in a Transformer) and ROME (Rank-One Model Editing). Our fork extends this code by additional evaluation scripts implementing the CounterFact+ benchmark. For installation instructions see the original repository.

### Installation
We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`.


### Running Experiments
See [INSTRUCTIONS.md](INSTRUCTIONS.md) for instructions on how to run the experiments and evaluations.

### How to Cite
#todo: update citation 
```bibtex
@article{hoelscher2023specificityplus,
  title={Detecting Edit Failures in LLMs: An Improved Specificity Benchmark},
  author={Jason Hoelscher-Obermaier and Persson, Julia, and Kran, Esben and Konstas, Ioannis and Barez, Fazl},
   month = may,
   year = "2023",
   address = "Toronto, canada",
  journal={Association for Computational Linguistics},
  url = "https://aclanthology.org/2023.findings-acl.0",
  
}
```

### Paper homepage
Find more information at https://specificityplus.apartresearch.com/.

