# Detecting Edit Failures in LLMs: An Improved Specificity Benchmark ([website](https://specificityplus.apartresearch.com/))

This repository contains the code for the paper [Detecting Edit Failures in LLMs: An Improved Specificity Benchmark](https://specificityplus.apartresearch.com/assets/CounterFact.pdf) (ACL Findings 2023).

It extends previous work on model editing by Meng et al. [[1]](#1) by introducing a new benchmark, called CounterFact+, for measuring the specificity of model edits.

### Attribution

The repository is a fork of [MEMIT](https://github.com/kmeng01/memit), which implements the model editing algorithms MEMIT (Mass Editing Memory in a Transformer) and ROME (Rank-One Model Editing). Our fork extends this code by additional evaluation scripts implementing the CounterFact+ benchmark. For installation instructions see the [original repository](https://github.com/kmeng01/memit).

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
@inproceedings{jason2023detecting,
title         = {Detecting Edit Failures In Large Language Models: An Improved Specificity Benchmark},
author        = {Hoelscher-Obermaier, Jason and Persson, Julia and Kran, Esben and Konstas, Ionnis and Barez, Fazl},
booktitle     = {Findings of ACL},
year          = {2023},
organization  = {Association for Computational Linguistics}
}
```

### Paper homepage

Find more information at https://specificityplus.apartresearch.com/.

### References

<a id="1">[1]</a>
Meng, Kevin, et al. "Mass-editing memory in a transformer." arXiv preprint arXiv:2210.07229 (2022).
