[![citation](http://img.shields.io/badge/Citation-0091FF.svg)](https://scholar.google.com/scholar?q=Sparsely%20Activated%20Networks.%20arXiv%202020)
[![arXiv](http://img.shields.io/badge/cs.LG-arXiv%3A1907.06592-B31B1B.svg)](https://arxiv.org/abs/1907.06592)

# Sparsely Activated Networks
This repository contains the code that generates the results of the paper **Sparsely Activated Networks** appeared in TNNLS.

## Requirements
- UNIX utilities (cmp, cp, echo, rm, touch)
- docker
- make
- nvidia-container-toolkit [required only when using CUDA]

## Instructions [more info on this template](https://github.com/pbizopoulos/cookiecutter-reproducible-builds-for-computational-research-papers)
1. `git clone https://github.com/pbizopoulos/sparsely-activated-networks`
2. `cd sparsely-activated-networks`
3. `sudo systemctl start docker`
4. make options
    * `make`             # Generate pdf.
    * `make ARGS=--full` # Generate full pdf.
    * `make clean`       # Remove cache, results directories and tex auxiliary files.
